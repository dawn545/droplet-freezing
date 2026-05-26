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

// 基础线性插值 (用于颜色混合)
static Float3 lerp(const Float3& c1, const Float3& c2, float t) {
    t = std::max(0.0f, std::min(1.0f, t));
    return {
        c1.r + (c2.r - c1.r) * t,
        c1.g + (c2.g - c1.g) * t,
        c1.b + (c2.b - c1.b) * t
    };
}

// 核心魔法：平滑阶跃函数 (Cubic Hermite)
// 它能将硬梆梆的线性边缘转化为完美的 S 型平滑曲线，彻底干掉毛边
static float smoothstep(float edge0, float edge1, float x) {
    float t = std::max(0.0f, std::min(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

// Catmull-Rom 三次插值权重 (a = -0.5)
// 与双线性相比提供 C1 连续，可消除斜向网格台阶
static inline void catmull_rom_weights(float t, float w[4]) {
    float t2 = t * t;
    float t3 = t2 * t;
    w[0] = -0.5f * t  + 1.0f * t2 - 0.5f * t3;
    w[1] =  1.0f      - 2.5f * t2 + 1.5f * t3;
    w[2] =  0.5f * t  + 2.0f * t2 - 1.5f * t3;
    w[3] =             -0.5f * t2 + 0.5f * t3;
}

static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// 双三次采样（Catmull-Rom）：在网格 (Nx, Ny) 上对连续坐标 (gx, gy) 求值
static inline float bicubic_sample(const std::vector<double>& field,
                                   int Nx, int Ny, float gx, float gy) {
    int ix = static_cast<int>(std::floor(gx));
    int iy = static_cast<int>(std::floor(gy));
    float tx = gx - ix;
    float ty = gy - iy;

    float wx[4], wy[4];
    catmull_rom_weights(tx, wx);
    catmull_rom_weights(ty, wy);

    float result = 0.0f;
    for (int j = 0; j < 4; ++j) {
        int y = clampi(iy - 1 + j, 0, Ny - 1);
        const double* row = &field[static_cast<size_t>(y) * Nx];
        float row_sum = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int x = clampi(ix - 1 + i, 0, Nx - 1);
            row_sum += wx[i] * static_cast<float>(row[x]);
        }
        result += wy[j] * row_sum;
    }
    return result;
}

// 每像素超采样阶数（SS x SS 子采样，2 => 4 样本/像素）
// 配合 Catmull-Rom 双三次插值，4 样本已足够给出干净的丝滑边缘
static constexpr int SS = 2;

static void update_pixels() {
    if (!solver) return;
    int Nx = solver->getNx();
    int Ny = solver->getNy();
    const auto& phi = solver->getPhi();
    const auto& fs = solver->getFs();

    if ((int)pixelBuffer.size() != windowWidth * windowHeight * 4) {
        pixelBuffer.assign(windowWidth * windowHeight * 4, 0);
    }

    // 颜色定义保持你的经典设定
    const Float3 colorGas = { 5.0f / 255.0f, 20.0f / 255.0f, 80.0f / 255.0f };
    const Float3 colorLiq = { 255.0f / 255.0f, 50.0f / 255.0f, 50.0f / 255.0f };
    const Float3 colorIce = { 220.0f / 255.0f, 220.0f / 255.0f, 220.0f / 255.0f };

    const float invW = 1.0f / static_cast<float>(windowWidth);
    const float invH = 1.0f / static_cast<float>(windowHeight);
    const float subStep = 1.0f / static_cast<float>(SS);
    const float subOffset = 0.5f * subStep;       // 子采样格的中心
    const float invSamples = 1.0f / static_cast<float>(SS * SS);

    for (int Y = 0; Y < windowHeight; ++Y) {
        for (int X = 0; X < windowWidth; ++X) {
            float accR = 0.0f, accG = 0.0f, accB = 0.0f;

            // SS x SS 子采样 + Catmull-Rom 双三次重建
            for (int sy = 0; sy < SS; ++sy) {
                float fY = (static_cast<float>(Y) + subOffset + sy * subStep) * invH;
                float gy = fY * (Ny - 1);

                for (int sx = 0; sx < SS; ++sx) {
                    float fX = (static_cast<float>(X) + subOffset + sx * subStep) * invW;
                    float gx = fX * (Nx - 1);

                    float phi_val = bicubic_sample(phi, Nx, Ny, gx, gy);
                    float fs_val  = bicubic_sample(fs,  Nx, Ny, gx, gy);

                    // 双三次会有微小越界，钳位到合法范围
                    if (phi_val < 0.0f) phi_val = 0.0f; else if (phi_val > 1.0f) phi_val = 1.0f;
                    if (fs_val  < 0.0f) fs_val  = 0.0f; else if (fs_val  > 1.0f) fs_val  = 1.0f;

                    float phi_t = smoothstep(0.42f, 0.58f, phi_val);
                    float solid_frac = phi_val * fs_val;
                    float fs_t = smoothstep(0.2f, 0.6f, solid_frac);

                    Float3 dropletColor = lerp(colorLiq, colorIce, fs_t);
                    Float3 finalColor = lerp(colorGas, dropletColor, phi_t);

                    accR += finalColor.r;
                    accG += finalColor.g;
                    accB += finalColor.b;
                }
            }

            accR *= invSamples;
            accG *= invSamples;
            accB *= invSamples;

            int pixel_id = Y * windowWidth + X;
            auto toByte = [](float v) -> uint8_t {
                if (v < 0.0f) v = 0.0f; else if (v > 1.0f) v = 1.0f;
                return static_cast<uint8_t>(v * 255.0f + 0.5f);
            };
            pixelBuffer[(pixel_id * 4) + 0] = toByte(accR);
            pixelBuffer[(pixel_id * 4) + 1] = toByte(accG);
            pixelBuffer[(pixel_id * 4) + 2] = toByte(accB);
            pixelBuffer[(pixel_id * 4) + 3] = 255;
        }
    }

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixelBuffer.data());
}

// === 下方的 initialize_rendering, display, timer_callback 保持不变 ===
void initialize_rendering(int argc, char** argv, int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("Phase-field LBM - Smoothstep Anti-Aliased");

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
