#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <tiny_obj_loader.h>
#include "gltf/tiny_gltf.h"
#include "gltf/stb_image.h"

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    scene_filename = filename;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadGeom(string objectid) {
    
    cout << "Loading Geom " << objectid << "..." << endl;
    Geom newGeom;
    string line;

    string mesh_filename;
    //load object type
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        if (strcmp(line.c_str(), "sphere") == 0) {
            cout << "Creating new sphere..." << endl;
            newGeom.type = SPHERE;
        }
        else if (strcmp(line.c_str(), "cube") == 0) {
            cout << "Creating new cube..." << endl;
            newGeom.type = CUBE;
        }
        else if (line.rfind("mesh", 0) == 0) {
            cout << "Loading new mesh..." << endl;
            newGeom.type = MESH;
            vector<string> tokens = utilityCore::tokenizeString(line);
            mesh_filename = tokens[1];
        }

    }

    //link material
    utilityCore::safeGetline(fp_in, line);
    if (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        //newGeom.materialid = atoi(tokens[1].c_str());
        newGeom.materialid = material_ids[tokens[1]];
        cout << "Connecting Geom " << objectid << " to Material " << tokens[1] << "..." << endl;
    }

    //load transformations
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);

        //load tranformations
        if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
            newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
            newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
            newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    newGeom.transform = utilityCore::buildTransformationMatrix(
        newGeom.translation, newGeom.rotation, newGeom.scale);
    newGeom.inverseTransform = glm::inverse(newGeom.transform);
    newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

    //load mesh
    string ext = mesh_filename.substr(mesh_filename.find_last_of(".") + 1);
    transform(ext.begin(), ext.end(), ext.begin(), tolower);
    if (ext == "obj")
    {
        loadObj(newGeom, mesh_filename);
    }
    else if (ext == "gltf") {
        loadGltf(newGeom, mesh_filename);
    }

    geoms.push_back(newGeom);
    return 1;
}

int Scene::loadGltf(Geom& newGeom, string filename) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    string scene_directory = scene_filename.substr(0, scene_filename.find_last_of("/\\") + 1);
    string obj_dirname = scene_directory + filename.substr(0, filename.find_last_of("/\\") + 1);
    string gltf_filename = scene_directory + filename;
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltf_filename.c_str());
    if (!warn.empty()) {
        std::cout << "GLTF load warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "GLTF load error: " << err << std::endl;
        return -1;
    }
    if (!ret) {
        std::cerr << "Failed to load GLTF file." << std::endl;
        return -1;
    }

    std::cout << "Loaded GLTF file successfully." << std::endl;

    // Process each mesh in the GLTF file
    for (const auto& mesh : model.meshes) {
        std::cout << "Processing mesh: " << mesh.name << std::endl;
        for (const auto& primitive : mesh.primitives) {
            //Put position data into vector
            int posAccessorIndex = primitive.attributes.at("POSITION");
            tinygltf::Accessor& posAccessor = model.accessors[posAccessorIndex];
            tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            tinygltf::Buffer& positionBuffer = model.buffers[posBufferView.buffer];
            const float* positions = reinterpret_cast<const float*>(&(positionBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]));
            int vStartIdx = vertices.size();
            for (size_t i = 0; i < posAccessor.count; i++) {
                glm::vec3 position(positions[i * 3 + 0], positions[i * 3 + 1], positions[i * 3 + 2]);
                position = glm::vec3(newGeom.transform * glm::vec4(position, 1.0f));
                vertices.push_back(position);
            }
            cout << "Read " << vertices.size() << " vertices." << endl;

            //Put normal data into vector
            int norAccessorIndex = primitive.attributes.at("NORMAL");
            tinygltf::Accessor& norAccessor = model.accessors[norAccessorIndex];
            tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
            tinygltf::Buffer& normalBuffer = model.buffers[norBufferView.buffer];
            const float* normalsPtr = reinterpret_cast<const float*>(&(normalBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset])); 
            int vnStartIdx = normals.size();
            for (size_t i = 0; i < norAccessor.count; i++) {
                glm::vec3 normal(normalsPtr[i * 3 + 0], normalsPtr[i * 3 + 1], normalsPtr[i * 3 + 2]);
                normal = glm::normalize(glm::vec3(newGeom.invTranspose * glm::vec4(normal, 0.0f))); 
                normals.push_back(normal);
            }
            cout << "Read " << normals.size() << " normals." << endl;

            //Put texcoord data into vector
            int vtStartIdx = texcoords.size();
            if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                int albedoAccessorIndex = primitive.attributes.at("TEXCOORD_0");
                tinygltf::Accessor& albedoAccessor = model.accessors[albedoAccessorIndex];
                tinygltf::BufferView& albedoBufferView = model.bufferViews[albedoAccessor.bufferView];
                tinygltf::Buffer& albedoBuffer = model.buffers[albedoBufferView.buffer];
                const float* uv = reinterpret_cast<const float*>(&(albedoBuffer.data[albedoBufferView.byteOffset + albedoAccessor.byteOffset]));
                for (size_t i = 0; i < albedoAccessor.count; ++i) {
                    glm::vec2 texcoord(uv[i * 2 + 0], uv[i * 2 + 1]);
                    texcoords.push_back(texcoord);
                }
                cout << "Read " << texcoords.size() - vtStartIdx << " texcoords for primitive." << endl;
            }
            else {
                cout << "TEXCOORD_0 attribute not found for primitive, skipping texcoords." << endl;
            }

            //Generate material
            if (primitive.material >= 0) {
                int index = model.materials[primitive.material].pbrMetallicRoughness.baseColorTexture.index;
                if (index != -1) {
                    tinygltf::Texture& texture = model.textures[index];
                    string albedoMapPath = "../assets/" + model.images[texture.source].uri;
                    cout << "Texture path: " << albedoMapPath << endl;
                }
                else {
                    cout << "No texture material for this mesh." << endl;
                }
            }
            else {
                cout << "No texture material for this mesh." << endl;
            }

            //Generate triangles
            if (primitive.indices >= 0) {
                tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                auto type = indexAccessor.componentType;
                if (indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT && indexAccessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    cout << "Unsupported index type for the give glTF model." << endl;
                    return -1;
                }
                newGeom.meshidx = meshes.size();
                uint32_t* indices = reinterpret_cast<uint32_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                for (size_t i = 0; i < indexAccessor.count; i += 3) {
                    Mesh newMesh;

                    newMesh.v[0] = indices[i] + vStartIdx;
                    newMesh.v[1] = indices[i + 1] + vStartIdx;
                    newMesh.v[2] = indices[i + 2] + vStartIdx;

                    if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
                        newMesh.vn[0] = indices[i] + vnStartIdx;
                        newMesh.vn[1] = indices[i + 1] + vnStartIdx;
                        newMesh.vn[2] = indices[i + 2] + vnStartIdx;
                    }

                    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
                        newMesh.vt[0] = indices[i] + vtStartIdx;
                        newMesh.vt[1] = indices[i + 1] + vtStartIdx;
                        newMesh.vt[2] = indices[i + 2] + vtStartIdx;
                    }
                    else 
                    {
                        newMesh.vt[0] = -1;
                        newMesh.vt[1] = -1;
                        newMesh.vt[2] = -1;
                    }

                    //TODO: Check the implementation
                    newMesh.materialid = newGeom.materialid;

                    newMesh.aabb.min = glm::min(vertices[newMesh.v[0]], glm::min(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
                    newMesh.aabb.max = glm::max(vertices[newMesh.v[0]], glm::max(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
                    newMesh.aabb.centroid = (newMesh.aabb.min + newMesh.aabb.max) * 0.5f;

                    meshes.push_back(newMesh);

                    
                }
                newGeom.meshcnt = meshes.size() - newGeom.meshidx;
                cout << "Number of meshes: " << newGeom.meshcnt << endl;
                newGeom.bvhrootidx = buildBVHEqualCount(newGeom.meshidx, newGeom.meshidx + newGeom.meshcnt);
            }
            
        }
    }
    return 1;
}


void Scene::loadObj(Geom& newGeom, string obj_filename)
{
    string scene_directory = scene_filename.substr(0, scene_filename.find_last_of("/\\") + 1);
    string obj_dirname = scene_directory + obj_filename.substr(0, obj_filename.find_last_of("/\\") + 1);
    obj_filename = scene_directory + obj_filename;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> tinyobj_materials;

    std::string err;
    bool ret = tinyobj::LoadObj(
        &attrib, &shapes, &tinyobj_materials, &err, obj_filename.c_str(), obj_dirname.c_str(), true);
    if (!err.empty())
    {
        std::cerr << err << std::endl;
    }
    if (!ret)
    {
        std::cerr << "Failed to load/parse .obj." << std::endl;
        exit(1);
    }

    int materialStartIdx = materials.size();
    // add materials
    if (tinyobj_materials.size() > 0)
    {
        for (const tinyobj::material_t& material : tinyobj_materials)
        {
            Material newMaterial;
            if (material.emission[0] + material.emission[1] + material.emission[2] > 0.0f)
            {
                newMaterial.albedo = glm::vec3(material.emission[0], material.emission[1], material.emission[2]);
                newMaterial.emittance = 1.0f;
            }
            else
            {
                newMaterial.albedo = glm::vec3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
                newMaterial.emittance = 0.0f;
            }
            newMaterial.metallic = material.metallic;
            newMaterial.roughness = material.roughness;
            newMaterial.ior = material.ior;
            newMaterial.opacity = material.dissolve;
            materials.push_back(newMaterial);
        }
    }

    // add vertices
    int vStartIdx = vertices.size();
    for (int i = 0; i < attrib.vertices.size() / 3; i++)
    {
        vertices.push_back(glm::vec3(
            newGeom.transform * glm::vec4(attrib.vertices[3 * i + 0],
                attrib.vertices[3 * i + 1],
                attrib.vertices[3 * i + 2], 1.0f)));
    }

    // add normals
    int vnStartIdx = normals.size();
    for (int i = 0; i < attrib.normals.size() / 3; i++)
    {
        normals.push_back(glm::normalize(glm::vec3(
            newGeom.transform * glm::vec4(attrib.normals[3 * i + 0],
                attrib.normals[3 * i + 1],
                attrib.normals[3 * i + 2], 0.0f))));
    }

    // add texcoords
    int vtStartIdx = texcoords.size();
    for (int i = 0; i < attrib.texcoords.size() / 2; i++)
    {
        texcoords.push_back(glm::vec2(attrib.texcoords[2 * i + 0],
            attrib.texcoords[2 * i + 1]));
    }

    // add meshes
    newGeom.meshidx = meshes.size();
    for (const tinyobj::shape_t& shape : shapes)
    {
        for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++)
        {
            const tinyobj::index_t& idx0 = shape.mesh.indices[3 * f + 0];
            const tinyobj::index_t& idx1 = shape.mesh.indices[3 * f + 1];
            const tinyobj::index_t& idx2 = shape.mesh.indices[3 * f + 2];

            Mesh newMesh;

            newMesh.v[0] = idx0.vertex_index + vStartIdx;
            newMesh.v[1] = idx1.vertex_index + vStartIdx;
            newMesh.v[2] = idx2.vertex_index + vStartIdx;

            newMesh.vn[0] = idx0.normal_index + vnStartIdx;
            newMesh.vn[1] = idx1.normal_index + vnStartIdx;
            newMesh.vn[2] = idx2.normal_index + vnStartIdx;

            newMesh.vt[0] = idx0.texcoord_index + vtStartIdx;
            newMesh.vt[1] = idx1.texcoord_index + vtStartIdx;
            newMesh.vt[2] = idx2.texcoord_index + vtStartIdx;

            newMesh.materialid = shape.mesh.material_ids[f] < 0
                ? newGeom.materialid
                : shape.mesh.material_ids[f] + materialStartIdx;

            // compute aabb
            newMesh.aabb.min = glm::min(vertices[newMesh.v[0]], glm::min(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
            newMesh.aabb.max = glm::max(vertices[newMesh.v[0]], glm::max(vertices[newMesh.v[1]], vertices[newMesh.v[2]]));
            newMesh.aabb.centroid = (newMesh.aabb.min + newMesh.aabb.max) * 0.5f;

            meshes.push_back(newMesh);
        }
    }
    newGeom.meshcnt = meshes.size() - newGeom.meshidx;

    // build bvh
    newGeom.bvhrootidx = buildBVHEqualCount(newGeom.meshidx, newGeom.meshidx + newGeom.meshcnt);

    cout << endl;
    cout << "Loaded " << obj_filename << endl;
    cout << "number of vertices: " << attrib.vertices.size() / 3 << endl;
    cout << "number of normals: " << attrib.normals.size() / 3 << endl;
    cout << "number of texcoords: " << attrib.texcoords.size() / 2 << endl;
    cout << "number of meshes: " << newGeom.meshcnt << endl;
    cout << "number of materials: " << tinyobj_materials.size() << endl;
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
            camera.lensRadius = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOCALDISTANCE") == 0) {
            camera.focalDistance = atof(tokens[1].c_str());
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    
    cout << "Loading Material " << materialid << "..." << endl;
    Material newMaterial;

    while (true) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (tokens.empty()) break;
        if (strcmp(tokens[0].c_str(), "RGB") == 0) {
            glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            newMaterial.color = glm::clamp(color, 0.0f, 1.0f);
            newMaterial.albedo = glm::clamp(color, 0.0f, 1.0f);
        }
        else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
            newMaterial.specular.exponent = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
            glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            newMaterial.specular.color = specColor;
        }
        else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
            newMaterial.hasReflective = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
            newMaterial.hasRefractive = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
            newMaterial.indexOfRefraction = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
            newMaterial.emittance = atof(tokens[1].c_str());
        }
        //PBR
        else if (strcmp(tokens[0].c_str(), "METALLIC") == 0) {
            newMaterial.metallic = glm::clamp((float)atof(tokens[1].c_str()), 0.0f, 1.0f);
        }
        else if (strcmp(tokens[0].c_str(), "ROUGHNESS") == 0) {
            newMaterial.roughness = glm::clamp((float)atof(tokens[1].c_str()), 0.0f, 1.0f);
        }
        else if (strcmp(tokens[0].c_str(), "IOR") == 0) {
            newMaterial.ior = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "OPACITY") == 0) {
            newMaterial.opacity = glm::clamp((float)atof(tokens[1].c_str()), 0.0f, 1.0f);
        }
        //Procedual
        else if (strcmp(tokens[0].c_str(), "PROC") == 0) {
            newMaterial.procTextType = atof(tokens[1].c_str());
            cout << "Procedual texture:" << newMaterial.procTextType << endl;
        }
        materials.push_back(newMaterial);
        material_ids[materialid] = materials.size() - 1;
    }
    return 1;
}

// reference https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies

int Scene::buildBVHEqualCount(int meshStartIdx, int meshEndIdx)
{
    // no mesh
    if (meshEndIdx == meshStartIdx)
    {
        return -1;
    }

    // FIXME: why can't emplace back
    // int nodeIdx = bvh.size();
    // bvh.emplace_back();
    // BVHNode& node = bvh.back();
    BVHNode node;

    // compute bvh aabb on CPU, expensive but only done once
    for (int i = meshStartIdx; i < meshEndIdx; i++)
    {
        node.aabb.min = glm::min(node.aabb.min, meshes[i].aabb.min);
        node.aabb.max = glm::max(node.aabb.max, meshes[i].aabb.max);
    }
    node.aabb.centroid = (node.aabb.min + node.aabb.max) * 0.5f;

    // one mesh, leaf node
    if (meshEndIdx - meshStartIdx == 1)
    {
        node.left = -1;
        node.right = -1;
        node.meshidx = meshStartIdx;
    }
    // multiple meshes, internal node
    else
    {
        // split method EqualCounts, range is [meshStartIdx, meshEndIdx)
        int mid = (meshStartIdx + meshEndIdx) / 2;
        glm::vec3 diff = node.aabb.max - node.aabb.min;
        int dim = (diff.x > diff.y && diff.x > diff.z) ? 0 : (diff.y > diff.z) ? 1 : 2;
        std::nth_element(meshes.begin() + meshStartIdx, meshes.begin() + mid, meshes.begin() + meshEndIdx,
            [dim](const Mesh& a, const Mesh& b) {
                return (a.aabb.centroid[dim] < b.aabb.centroid[dim]);
            }
        );

        node.left = buildBVHEqualCount(meshStartIdx, mid);
        node.right = buildBVHEqualCount(mid, meshEndIdx);
        node.meshidx = -1;
    }

    // FIXME: why can't emplace back
    // return nodeIdx;
    bvh.push_back(node);
    return bvh.size() - 1;
}