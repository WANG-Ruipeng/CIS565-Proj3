#pragma once

#include <vector>
#include "scene.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void showGBuffer(uchar4* pbo, bool showPosition);
void showImage(uchar4* pbo, int iter);
void showDenoised(uchar4* pbo, int filter_size, float c_phi, float n_phi, float p_phi, int iter);