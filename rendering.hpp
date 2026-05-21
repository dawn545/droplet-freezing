#pragma once
#include "physics.hpp"

extern LBMSolver* solver;          // 全局求解器指针（由 main 设置）
void initialize_rendering(int argc, char** argv, int width, int height);
void display();                     // GLUT 显示回调
void timer_callback(int value);    // 定时器回调，驱动时间步进
