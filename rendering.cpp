#include "rendering.hpp"
#include <GL/freeglut.h>
#include <algorithm>
#include <vector>
#include <cstdint>

extern LBMSolver* solver;

static std::vector<uint8_t> pixelBuffer;
static int windowWidth = 800;
static int windowHeight = 400;
static GLuint textureID;

struct Float3 { float r, g, b; };

// 重新引入高效的线性插值工具
static Float3 lerp(const Float3& c1, const Float3& c2, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    return {
        c1.r + (c2.r - c1.r) * t,
        c1.g + (c2.g - c1.g) * t,
        c1.b + (c2.b - c1.b) * t
    };
}

static void update_pixels() {
    if (!solver) return;
    int Nx = solver->getNx();
    int Ny = solver->getNy();
    const auto& phi = solver->getPhi();
    const auto& fs = solver->getFs();

    if ((int)pixelBuffer.size() != Nx * Ny * 4) {
        pixelBuffer.assign(Nx * Ny * 4, 0);
    }

    // 颜色配置（保持高对比度）
    Float3 colorGas = { 5.0f / 255.0f, 20.0f / 255.0f, 80.0f / 255.0f };   // 外部气相 (深蓝)
    Float3 colorLiq = { 255.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f }; // 液态水 (红)
    Float3 colorIce = { 210.0f / 255.0f, 240.0f / 255.0f, 255.0f / 255.0f };// 固态冰 (灰白)

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = y * Nx + x;
            float phi_val = static_cast<float>(phi[id]); 
            float fs_val = static_cast<float>(fs[id]);   

            // 1. 解决冰边界“分节、粗糙”问题
            // 物理场中的 fs 往往是阶跃式的块状，我们用稍宽的窗口 [0.2, 0.8] 将断裂的网格无缝融合成平滑的晶面
            float fs_t = (fs_val - 0.2f) / (0.8f - 0.2f);
            fs_t = std::max(0.0f, std::min(1.0f, fs_t));
            Float3 dropletColor = lerp(colorLiq, colorIce, fs_t);

            // 2. 解决外边界“大范围发晕”与“硬切大锯齿”的矛盾
            // 采用极窄通道 [0.46, 0.54]。这在屏幕上只占 1~2 像素过渡
            // 既能保证边界看起来清晰锐利（没有晕圈），又实现了完美的曲线抗锯齿（消除方块感）
            float phi_t = (phi_val - 0.46f) / (0.54f - 0.46f);
            phi_t = std::max(0.0f, std::min(1.0f, phi_t));
            Float3 finalColor = lerp(colorGas, dropletColor, phi_t);

            // 3. 数据打包
            auto toByte = [](float v) -> uint8_t { return static_cast<uint8_t>(v * 255.0f); };
            pixelBuffer[(id * 4) + 0] = toByte(finalColor.r);
            pixelBuffer[(id * 4) + 1] = toByte(finalColor.g);
            pixelBuffer[(id * 4) + 2] = toByte(finalColor.b);
            pixelBuffer[(id * 4) + 3] = 255; 
        }
    }

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Nx, Ny, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer.data());
}

void initialize_rendering(int argc, char** argv, int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Phase-field LBM - Subpixel Anti-Aliased Rendering");

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glutDisplayFunc(display);
    glutTimerFunc(16, timer_callback, 0);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    update_pixels();

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glColor3f(1.0f, 1.0f, 1.0f);

    glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(static_cast<float>(windowWidth), 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(static_cast<float>(windowWidth), static_cast<float>(windowHeight));
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, static_cast<float>(windowHeight));
    glEnd();

    glutSwapBuffers();
}

void timer_callback(int value) {
    if (solver) solver->step(20);
    glutPostRedisplay();
    glutTimerFunc(16, timer_callback, 0);
}
