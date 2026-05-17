#include "rendering.hpp"
#include <GL/freeglut.h>
#include <algorithm>
#include <vector>
#include <cstdint>

extern LBMSolver* solver;
static std::vector<uint8_t> pixelBuffer;
static int windowWidth = 800;
static int windowHeight = 400;

static void update_pixels() {
    if (!solver) return;
    int Nx = solver->getNx();
    int Ny = solver->getNy();
    const auto& phi = solver->getPhi();
    const auto& fs = solver->getFs();
    if ((int)pixelBuffer.size() != Nx * Ny * 3)
        pixelBuffer.assign(Nx * Ny * 3, 0);

    auto isSolidLiquidInterface = [&](int x, int y) {
        if (y < 0 || y >= Ny || x < 0 || x >= Nx) return false;
        int id = y * Nx + x;
        if (phi[id] <= 0.5) return false;
        bool solid = fs[id] > 0.5;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                int ny = y + dy;
                if (nx < 0 || nx >= Nx || ny < 0 || ny >= Ny) continue;
                int nid = ny * Nx + nx;
                if (phi[nid] > 0.5 && (fs[nid] > 0.5) != solid) {
                    return true;
                }
            }
        }
        return false;
    };

    // 在 update_pixels 函数内的双重循环中替换原有的颜色判断逻辑：
    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
        int id = y * Nx + x;
        double phi_val = phi[id];
        double fs_val = fs[id]; // 获取当前的固相分数
        
        uint8_t r, g, b;

        if (phi_val > 0.5) {
            // 液滴内部：通过固相分数 fs_val 区分水(fs=0)和冰(fs=1)
            // 水的颜色 (原液相颜色)
            uint8_t water_r = 15, water_g = 170, water_b = 230;
            // 冰的颜色 (通常为更亮的浅蓝色或灰白色)
            uint8_t ice_r = 210, ice_g = 240, ice_b = 255; 

            // 限制 fs_val 在 0.0 到 1.0 之间，防止颜色溢出
            double fs_clamped = std::max(0.0, std::min(1.0, fs_val));

            // 对冰和水进行线性颜色插值，形成平滑的冻结前沿
            r = water_r + fs_clamped * (ice_r - water_r);
            g = water_g + fs_clamped * (ice_g - water_g);
            b = water_b + fs_clamped * (ice_b - water_b);
        } else {
            // 气相
            r = 5; g = 20; b = 80;
        }

        pixelBuffer[(id * 3) + 0] = r;
        pixelBuffer[(id * 3) + 1] = g;
        pixelBuffer[(id * 3) + 2] = b;
    }
    }
}

void initialize_rendering(int argc, char** argv, int width, int height) {
    windowWidth = width;
    windowHeight = height;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("Phase-field LBM - droplet on ground");
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClearColor(0.95f, 0.95f, 0.98f, 1.0f);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, windowWidth, 0, windowHeight, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutDisplayFunc(display);
    glutTimerFunc(16, timer_callback, 0);
}

void display() {
    if (!solver) return;
    update_pixels();
    int Nx = solver->getNx();
    int Ny = solver->getNy();

    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(0, 0);
    glPixelZoom((float)windowWidth / Nx, (float)windowHeight / Ny);
    glDrawPixels(Nx, Ny, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer.data());
    glutSwapBuffers();
}

void timer_callback(int value) {
    if (!solver) return;
    solver->step(3);
    glutPostRedisplay();
    glutTimerFunc(16, timer_callback, 0);
}
