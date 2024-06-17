#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    string scene_filename;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadGltf(Geom& gltfMesh, string filename);
    int loadCamera();
    void loadObj(Geom& newGeom, string obj_filename);
    int buildBVHEqualCount(int meshStartIdx, int meshEndIdx);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::unordered_map<string, int> material_ids;
    RenderState state;

    //Obj Mesh
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;  
    std::vector<glm::vec2> texcoords;
    std::vector<Mesh> meshes;
    std::vector<BVHNode> bvh;
};
