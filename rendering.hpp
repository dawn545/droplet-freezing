#pragma once
#include "physics.hpp"

extern LBMSolver* solver;
void initialize_rendering(int argc, char** argv, int width, int height);
void display();
void timer_callback(int value);
