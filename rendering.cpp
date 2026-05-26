#include "rendering.hpp"
#include <GL/freeglut.h>
#include <algorithm>
#include <vector>
#include <cstdint>

extern LBMSolver* solver;

// 使用 RGBA (4通道) 替代 RGB (3通道)，以获得更好的 GPU 内存对齐性能
static std::vector<uint8_t> pixelBuffer;
static int windowWidth = 800;
static int windowHeight = 400;

// OpenGL 纹理 ID
static GLuint textureID;

// --- 辅助工具：三维向量与线性插值 ---
struct Float3 { float r, g, b; };

// 线性插值函数 (Lerp): 根据比例 t 在颜色 c1 和 c2 之间平滑过渡
static Float3 lerp(const Float3& c1, const Float3& c2, float t) {
    // 限制 t 的范围在 [0.0, 1.0] 之间，防止颜色溢出
    t = std::max(0.0f, std::min(1.0f, t));
    return {
        c1.r + (c2.r - c1.r) * t,
        c1.g + (c2.g - c1.g) * t,
        c1.b + (c2.b - c1.b) * t
    };
}

// --- 核心更新逻辑：物理场转颜色纹理 ---
static void update_pixels() {
    if (!solver) return;
    int Nx = solver->getNx();
    int Ny = solver->getNy();
    const auto& phi = solver->getPhi();
    const auto& fs = solver->getFs();

    // 重新分配内存为 4 通道 (RGBA)
    if ((int)pixelBuffer.size() != Nx * Ny * 4) {
        pixelBuffer.assign(Nx * Ny * 4, 0);
    }

    // 定义基础颜色 (归一化为 0.0 ~ 1.0 范围，便于插值)
    Float3 colorGas = { 5.0f / 255.0f, 20.0f / 255.0f, 80.0f / 255.0f };   // 外部气相 (深蓝)
    Float3 colorLiq = { 255.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f }; // 液态水 (红)
    Float3 colorIce = { 210.0f / 255.0f, 240.0f / 255.0f, 255.0f / 255.0f };// 固态冰 (灰白)

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = y * Nx + x;
            float phi_val = static_cast<float>(phi[id]); // 相场: 0=气, 1=液滴(包含水/冰)
            float fs_val = static_cast<float>(fs[id]);   // 固相分数: 0=液态水, 1=纯冰

            // 1. 计算液滴内部的颜色 (水和冰混合)
            // 根据 fs_val 在红色(水)和灰白(冰)之间平滑过渡
            Float3 dropletColor = lerp(colorLiq, colorIce, fs_val);

            // 2. 计算最终颜色 (气相和液滴混合)
            // 根据 phi_val 在深蓝(气体)和混合后的液滴颜色之间平滑过渡
            Float3 finalColor = lerp(colorGas, dropletColor, phi_val);

            // 3. 浮点转字节并写入 Buffer
            auto toByte = [](float v) -> uint8_t { return static_cast<uint8_t>(v * 255.0f); };

            pixelBuffer[(id * 4) + 0] = toByte(finalColor.r);
            pixelBuffer[(id * 4) + 1] = toByte(finalColor.g);
            pixelBuffer[(id * 4) + 2] = toByte(finalColor.b);
            pixelBuffer[(id * 4) + 3] = 255; // Alpha 不透明
        }
    }

    // 将更新后的像素数组推送到 GPU 纹理
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Nx, Ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer.data());
}

// --- 初始化 OpenGL 与 GLUT ---
void initialize_rendering(int argc, char** argv, int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Phase-field LBM - Optimized Hardware Rendering");

    // 配置正交投影矩阵 (左下角为 0,0，右上角为 width, height)
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // === 新增：纹理初始化 ===
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    // 设置双线性过滤，放大画面时边缘会平滑过渡，消除马赛克感
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 防止纹理边缘出现杂色
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    
    glutDisplayFunc(display);
    glutTimerFunc(16, timer_callback, 0);
}

// --- 渲染画面 ---
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // 计算颜色并推送到 GPU 纹理
    update_pixels();

    // 绑定纹理并设置绘制颜色为纯白（避免染色）
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glColor3f(1.0f, 1.0f, 1.0f);

    // === 新增：使用硬件光栅化绘制全屏四边形 ===
    // 将物理网格纹理铺满整个窗口
    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(static_cast<float>(windowWidth), 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(static_cast<float>(windowWidth), static_cast<float>(windowHeight));
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, static_cast<float>(windowHeight));
    glEnd();

    glutSwapBuffers();
}

// --- 时间步进回调不变 ---
void timer_callback(int value) {
    if (solver) {
        // 推进20个物理步
        solver->step(20);
    }
    glutPostRedisplay();
    glutTimerFunc(16, timer_callback, 0);
}
