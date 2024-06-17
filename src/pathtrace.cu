#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_MATERIAL_ID 1
#define CACHE 1
#define BVH 1
#define DEPTH_OF_FIELD 1
#define OIDN 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static ShadeableIntersection* dev_cache = NULL;
static glm::vec3* dev_vertices = NULL;
static Mesh* dev_meshes = NULL;
static glm::vec3* dev_normals = NULL;
static glm::vec2* dev_texcoords = NULL;

#if BVH
static BVHNode* dev_bvh = NULL;
#endif

#if OIDN
#define EMA_ALPHA 0.2f
#define DENOISE_INTERVAL 100

#include "OpenImageDenoise/oidn.hpp"

static glm::vec3* dev_denoised = NULL;
static glm::vec3* dev_albedo = NULL;
static glm::vec3* dev_normal = NULL;

void denoise()
{
	int width = hst_scene->state.camera.resolution.x,
		height = hst_scene->state.camera.resolution.y;

	// Create an Intel Open Image Denoise device
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();

	// Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
	// oidn::BufferRef colorBuf = device.newBuffer(width * height * sizeof(glm::vec3));
	// oidn::BufferRef albedoBuf = device.newBuffer(width * height * sizeof(glm::vec3));
	// oidn::BufferRef normalBuf = device.newBuffer(width * height * sizeof(glm::vec3));

	// Create a filter for denoising a beauty (color) image using prefiltered auxiliary images too
	oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
	filter.setImage("color", dev_image, oidn::Format::Float3, width, height); // beauty
	filter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height); // auxiliary
	filter.setImage("normal", dev_normal, oidn::Format::Float3, width, height); // auxiliary
	filter.setImage("output", dev_denoised, oidn::Format::Float3, width, height); // denoised beauty
	filter.set("hdr", true); // image is HDR
	filter.set("cleanAux", true); // auxiliary images will be prefiltered
	filter.commit();

	// Copy the rendered image to the color buffer
	// cudaMemcpy(colorBuf.getData(), dev_image, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	// cudaMemcpy(albedoBuf.getData(), dev_albedo, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	// cudaMemcpy(normalBuf.getData(), dev_normal, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	// cudaDeviceSynchronize();

	// Create a separate filter for denoising an auxiliary albedo image (in-place)
	oidn::FilterRef albedoFilter = device.newFilter("RT"); // same filter type as for beauty
	albedoFilter.setImage("albedo", dev_albedo, oidn::Format::Float3, width, height);
	albedoFilter.setImage("output", dev_albedo, oidn::Format::Float3, width, height);
	albedoFilter.commit();

	// Create a separate filter for denoising an auxiliary normal image (in-place)
	oidn::FilterRef normalFilter = device.newFilter("RT"); // same filter type as for beauty
	normalFilter.setImage("normal", dev_normal, oidn::Format::Float3, width, height);
	normalFilter.setImage("output", dev_normal, oidn::Format::Float3, width, height);
	normalFilter.commit();

	// Prefilter the auxiliary images
	albedoFilter.execute();
	normalFilter.execute();

	// Filter the beauty image
	filter.execute();

	// Check for errors
	const char* errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None)
		std::cout << "Error: " << errorMessage << std::endl;
}

__global__
void copyFirstTraceResult(
	PathSegment* pathSegments, int num_paths,
	ShadeableIntersection* shadeableIntersections,
	glm::vec3* albedo, glm::vec3* normal)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		PathSegment pathSegment = pathSegments[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];

		albedo[pathSegment.pixelIndex] = pathSegment.color;
		normal[pathSegment.pixelIndex] = intersection.surfaceNormal;
	}
}

__global__
void emaMergeDenoisedAndImage(int pixelcount, glm::vec3* image, glm::vec3* denoised)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < pixelcount) {
		// exponential moving average
		image[idx] = image[idx] * (1 - EMA_ALPHA) + denoised[idx] * EMA_ALPHA;
	}
}
#endif // OIDN

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_meshes, scene->meshes.size() * sizeof(Mesh));
	cudaMemcpy(dev_meshes, scene->meshes.data(), scene->meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice);
	
#if BVH
	cudaMalloc(&dev_bvh, scene->bvh.size() * sizeof(BVHNode));
	cudaMemcpy(dev_bvh, scene->bvh.data(), scene->bvh.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
#endif

	cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
	cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_texcoords, scene->texcoords.size() * sizeof(glm::vec2));
	cudaMemcpy(dev_texcoords, scene->texcoords.data(), scene->texcoords.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

#if OIDN
	cudaMalloc(&dev_denoised, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_normal, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_normal, 0, pixelcount * sizeof(glm::vec3));
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_cache);
	cudaFree(dev_vertices);
	cudaFree(dev_meshes);

#if BVH
	cudaFree(dev_bvh);
#endif

#if OIDN
	cudaFree(dev_denoised);
	cudaFree(dev_albedo);
	cudaFree(dev_normal);
#endif

	cudaFree(dev_normals);
	cudaFree(dev_texcoords);

	checkCUDAError("pathtraceFree");
}

//Procedual Texture
__device__ glm::vec3 checkerboard(glm::vec2 uv)
{
	if ((int)(uv.x * 10) % 2 == (int)(uv.y * 10) % 2)
		return glm::vec3(.2f);
	else
		return glm::vec3(.8f);
}
__device__ glm::vec3 palettes(glm::vec2 uv)
{
	glm::vec3 a(0.5, 0.5, 0.5), b(0.5, 0.5, 0.5), c(1.0, 1.0, 1.0), d(0.00, 0.33, 0.67);
	return a + b * glm::cos(TWO_PI * (c * glm::length(uv) + d));
}

#if DEPTH_OF_FIELD
__host__ __device__
glm::vec2 ConcentricSampleDisk(const glm::vec2& u)
{
	glm::vec2 uOffset = 2.0f * u - glm::vec2(1.0f, 1.0f);

	if (uOffset.x == 0.0f && uOffset.y == 0.0f)
	{
		return glm::vec2(0.0f, 0.0f);
	}

	float theta, r;
	if (glm::abs(uOffset.x) > glm::abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI_OVER_FOUR * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI_OVER_TWO - PI_OVER_FOUR * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}
#endif

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + u01(rng) - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + u01(rng) - (float)cam.resolution.y * 0.5f)
		);
#if DEPTH_OF_FIELD
		if (cam.lensRadius > 0)
		{
			glm::vec2 pLens = cam.lensRadius * ConcentricSampleDisk(glm::vec2(u01(rng), u01(rng)));
			float ft = cam.focalDistance / glm::dot(cam.view, segment.ray.direction);
			glm::vec3 pFocus = segment.ray.origin + segment.ray.direction * ft;
			segment.ray.origin += cam.right * pLens.x + cam.up * pLens.y;
			segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
		}
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment segment = pathSegments[idx];
		/*if(segment.remainingBounces <= 0)
		{
			return;
		}*/
		if (intersection.t > 0.0f) { // if the intersection exists...
			//segment.remainingBounces--;
			// Set up the RNG
			// LOOK: this is how you use thrust's RNG! Please look at makeSeededRandomEngine as well.
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				segment.color *= (materialColor * material.emittance);
				segment.remainingBounces = 0;
			}
			// Otherwise, do some pseudo-lighting computation. This is actually more
			// like what you would expect from shading in a rasterizer like OpenGL.
			// TODO: replace this! you should be able to start with basically a one-liner
			else {
				switch (material.procTextType){
					case 1: material.color = checkerboard(intersection.texcoord); break;
					case 2: material.color = palettes(intersection.texcoord); break;
					default: break;
				}
				glm::vec3 intersect = intersection.t * segment.ray.direction + segment.ray.origin;
				scatterRay(segment, intersect, intersection.surfaceNormal, material, rng);
				if (pathSegments[idx].remainingBounces == 0) pathSegments[idx].color = glm::vec3(0.0f);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			segment.color = glm::vec3(0.0f);
			segment.remainingBounces = 0;
		}
		pathSegments[idx] = segment;
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

// Stream Compaction Bool Judge
struct isRayAlive
{
	__host__ __device__ bool operator()(const PathSegment &path)
	{
		return path.remainingBounces > 0;
	}
};

// Sort
struct materialsCmp
{
	__host__ __device__ bool operator()(const ShadeableIntersection& m1,
	                                    const ShadeableIntersection& m2)
	{
		return m1.materialId < m2.materialId;
	}
};

// reference https://tavianator.com/2011/ray_box.html
__host__ __device__ bool aabbIntersectionTest(const AABB& aabb, const Ray& ray)
{
	float invDirX = 1.f / ray.direction.x;
	float invDirY = 1.f / ray.direction.y;
	float invDirZ = 1.f / ray.direction.z;

	float tx1 = (aabb.min.x - ray.origin.x) * invDirX;
	float tx2 = (aabb.max.x - ray.origin.x) * invDirX;

	float tmin = glm::min(tx1, tx2);
	float tmax = glm::max(tx1, tx2);

	float ty1 = (aabb.min.y - ray.origin.y) * invDirY;
	float ty2 = (aabb.max.y - ray.origin.y) * invDirY;

	tmin = glm::max(tmin, glm::min(ty1, ty2));
	tmax = glm::min(tmax, glm::max(ty1, ty2));

	float tz1 = (aabb.min.z - ray.origin.z) * invDirZ;
	float tz2 = (aabb.max.z - ray.origin.z) * invDirZ;

	tmin = glm::max(tmin, glm::min(tz1, tz2));
	tmax = glm::min(tmax, glm::max(tz1, tz2));

	return tmax >= glm::max(0.f, tmin);
}

__host__ __device__ void finalIntersectionTest(
	const Mesh& m, const glm::vec3* vertices, const glm::vec3* normals, const glm::vec2* texcoords,
	const Ray& r,
	float& t_min, glm::vec3& intersectionPoint, int& materialid, glm::vec3& normal, glm::vec2& texcoord)
{
	glm::vec3 v0 = vertices[m.v[0]];
	glm::vec3 v1 = vertices[m.v[1]];
	glm::vec3 v2 = vertices[m.v[2]];

	glm::vec3 baryPosition;

	if (glm::intersectRayTriangle(r.origin, r.direction, v0, v1, v2, baryPosition))
	{
		glm::vec3 point = (1 - baryPosition.x - baryPosition.y) * v0 +
			baryPosition.x * v1 +
			baryPosition.y * v2;
		float t = glm::length(r.origin - point);
		if (t > 0.0f && t < t_min)
		{
			t_min = t;
			intersectionPoint = point;
			materialid = m.materialid;
			if (m.vn[0] == -1 || m.vn[1] == -1 || m.vn[2] == -1)
				normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
			else
				normal = (1 - baryPosition.x - baryPosition.y) * normals[m.vn[0]] +
				baryPosition.x * normals[m.vn[1]] +
				baryPosition.y * normals[m.vn[2]];
			if (m.vt[0] == -1 || m.vt[1] == -1 || m.vt[2] == -1)
				texcoord = glm::vec2(-1.f); // no texture
			else
				texcoord = (1 - baryPosition.x - baryPosition.y) * texcoords[m.vt[0]] +
				baryPosition.x * texcoords[m.vt[1]] +
				baryPosition.y * texcoords[m.vt[2]];
		}
	}
}

// reference https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/
__host__ __device__ float meshIntersectionTestBVH(
	Geom& geom,
	BVHNode* bvh,
	Mesh* meshes, glm::vec3* vertices, glm::vec3* normals, glm::vec2* texcoords,
	Ray r,
	glm::vec3& intersectionPoint, int& materialid, glm::vec3& normal, glm::vec2& texcoord)
{
	float t_min = FLT_MAX;

	int stack[64];
	int* stackPtr = stack;
	*stackPtr++ = -1;

	int nodeIdx = geom.bvhrootidx;

	do
	{
		BVHNode node = bvh[nodeIdx];

		int idxL = node.left;
		BVHNode childL = bvh[idxL];
		int idxR = node.right;
		BVHNode childR = bvh[idxR];

		bool overlapL = aabbIntersectionTest(childL.aabb, r);
		bool overlapR = aabbIntersectionTest(childR.aabb, r);

		if (overlapL && (childL.left == -1 && childL.right == -1))
		{
			finalIntersectionTest(
				meshes[childL.meshidx], vertices, normals, texcoords,
				r,
				t_min, intersectionPoint, materialid, normal, texcoord);
		}

		if (overlapR && (childR.left == -1 && childR.right == -1))
		{
			finalIntersectionTest(
				meshes[childR.meshidx], vertices, normals, texcoords,
				r,
				t_min, intersectionPoint, materialid, normal, texcoord);
		}

		bool traverseL = overlapL && !(childL.left == -1 && childL.right == -1);
		bool traverseR = overlapR && !(childR.left == -1 && childR.right == -1);

		if (!traverseL && !traverseR)
			nodeIdx = *--stackPtr;
		else
		{
			nodeIdx = (traverseL) ? idxL : idxR;
			if (traverseL && traverseR)
				*stackPtr++ = idxR;
		}

	} while (nodeIdx != -1);

	return t_min;
}

// TODO:
// computeIntersectionsCache handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int iter
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
	//For cache
	, ShadeableIntersection* cache
	//For BVH Acceleration
#if BVH
	, BVHNode* bvh
#endif
	, Mesh* meshes
	, glm::vec3* vertices
	, glm::vec3* normals
	, glm::vec2* texcoords
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	bool toCache = CACHE && depth == 0;
	if (path_index < num_paths)
	{
		if (toCache && iter > 1)
		{
			intersections[path_index] = cache[path_index];
			return;
		}
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		glm::vec2 texcoord;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		int tmp_material_index;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_texcoord;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				tmp_normal = outside ? tmp_normal : -tmp_normal;
				tmp_texcoord = glm::vec2(-1.f);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				tmp_normal = outside ? tmp_normal : -tmp_normal;
				tmp_texcoord = glm::vec2(-1.f);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
			else if (geom.type == MESH)
			{
#if BVH
				t = meshIntersectionTestBVH(
					geom,
					bvh,
					meshes, vertices, normals, texcoords,
					pathSegment.ray,
					tmp_intersect, tmp_material_index, tmp_normal, tmp_texcoord);
#endif
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				texcoord = tmp_texcoord;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].texcoord = texcoord;
			if (toCache && iter == 1) cache[path_index] = intersections[path_index];
		}
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, iter, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size()
			, dev_intersections, dev_cache
#if BVH
			, dev_bvh
#endif 
			, dev_meshes, dev_vertices, dev_normals, dev_texcoords
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		// 按照材质id排序，实测中，简单场景甚至负优化，但复杂的场景这样做的效果还可以
#ifndef SORT_MATERIAL_ID
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialsCmp());
#endif

		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > 
			(iter, num_paths, dev_intersections, dev_paths, dev_materials);

#if OIDN
		// copy image to albedo and normal
		if (depth == 1 && (iter % DENOISE_INTERVAL == 0 || iter == hst_scene->state.iterations))
			copyFirstTraceResult << <numblocksPathSegmentTracing, blockSize1d >> > (
				dev_paths, num_paths, dev_intersections, dev_albedo, dev_normal);
#endif

#ifdef STREAM_COMPACTION
		// partition将isRayAlive的值为真的元素放到数组前面，不满足的放到后面。
		// 最后返回第一个不满足的元素指针，相减就是剩下还有迭代次数的光线数量了
		num_paths = thrust::partition(thrust::device,
		                              dev_paths, dev_paths + num_paths, isRayAlive()) - dev_paths;
#endif
		iterationComplete = depth == traceDepth || num_paths == 0;
		// TODO: should be based off stream compaction results.

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

#ifdef STREAM_COMPACTION
	// 记得恢复光线的数量，不然图像就黑了。
	// 因为流压缩的存在，最后num_paths会等于0，不恢复的话后面的迭代都不会有光线射出来了
	num_paths = dev_path_end - dev_paths;
#endif

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

#if OIDN
	if (iter % DENOISE_INTERVAL == 0)
	{
		denoise();
		emaMergeDenoisedAndImage << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_denoised);
	}
	else if (iter == hst_scene->state.iterations)
	{
		denoise();
		std::swap(dev_image, dev_denoised);
	}
#endif

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}