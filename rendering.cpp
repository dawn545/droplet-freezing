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

    for (int y = 0; y < Ny; ++y) {
        for (int x = 0; x < Nx; ++x) {
            int id = y * Nx + x;
            double phi_val = phi[id];
            double fs_val = fs[id]; 
            
            uint8_t r, g, b;
            if (phi_val > 0.5) {
                // 液相内部：基于固相结冰率 fs_val 在水和冰的颜色之间实施平滑插值
                uint8_t water_r = 15,  water_g = 170, water_b = 230; // 蔚蓝色水滴
                uint8_t ice_r = 210,   ice_g = 240,   ice_b = 255;   // 灰白色冰晶
                
                double fs_clamped = std::clamp(fs_val, 0.0, 1.0);
                r = water_r + fs_clamped * (ice_r - water_r);
                g = water_g + fs_clamped * (ice_g - water_g);
                b = water_b + fs_clamped * (ice_b - water_b);
            } else {
                // 气相/环境颜色
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
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // 激活 GLUT_DOUBLE 硬件双缓冲
    glutInitWindowSize(width, height);
    glutCreateWindow("Phase-field LBM - Droplet Solidification Freezing Simulation");
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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
    
    // 先清理前台颜色缓冲区，随后向后台缓冲区写数据，可杜绝绘制时产生的闪烁
    glClear(GL_COLOR_BUFFER_BIT); 
    
    update_pixels();
    int Nx = solver->getNx();
    int Ny = solver->getNy();

    glRasterPos2i(0, 0);
    glPixelZoom((float)windowWidth / Nx, (float)windowHeight / Ny);
    glDrawPixels(Nx, Ny, GL_RGB, GL_UNSIGNED_BYTE, pixelBuffer.data());
    
    glutSwapBuffers(); // 垂直同步安全调换前后端缓冲区
}

void timer_callback(int value) {
    if (!solver) return;
    solver->step(4); // 每个时钟周期向前安全迭代 4 个晶格步
    glutPostRedisplay();
    glutTimerFunc(16, timer_callback, 0);
}
