#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};

class BVHNode {
private:
	BVHNode* left;
	BVHNode* right;
	AABB bounding_box;
public:
	BVHNode();
	~BVHNode();

	bool createBoundingBox(glm::vec3 min, glm::vec3 max);
};

class BVHTree {
private:
	BVHNode* root;
	// BVH树的数组形式
	std::vector<int> node_list;

public:
	BVHTree();
	~BVHTree();
};

