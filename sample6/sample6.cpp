
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/*
  This is a hacked version of sample6.cpp, not using any of the existing mesh classes, to
  understand what the minimum amount of work needed to render a scene with multiple materials
  and meshes is.

  I did this after looking over the existing Mesh/OptixMesh code, and just feeling that it
  was way to spread out for me to actually grok what was going on.

  NB: This sample uses the one_bounce_diffuse program, so various bits and pieces that aren't
  needed for this have been ripped out.
*/

#include "MeshScene.h"
#include "commonStructs.h"
#include "random.h"
#include <GLUTDisplay.h>
#include <sutil.h>

#include <ImageLoader.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace optix;

#include <stdint.h>
#include <vector>
#include <windows.h>
using namespace std;
typedef uint32_t u32;

#include "boba_scene_format.hpp"

enum RayTypes
{
  RT_Radiance,
  RT_Shadow,
  RT_COUNT,
};

//------------------------------------------------------------------------------
//
// MeshViewer class
//
//------------------------------------------------------------------------------
struct MeshViewer : public MeshScene
{
  MeshViewer();

  // From SampleScene
  virtual void initScene(InitialCameraData& camera_data);
  virtual void doResize(unsigned int width, unsigned int height);
  virtual void trace(const RayGenCameraData& camera_data);
  virtual void cleanUp();
  virtual bool keyPressed(unsigned char key, int x, int y);
  virtual Buffer getOutputBuffer();

  void initContext();
  void initLights();
  void initGeometry();
  void initCamera(InitialCameraData& cam_data);
  void preprocess();
  void resetAccumulation();
  void genRndSeeds(unsigned int width, unsigned int height);

  bool Load(const char* filename);
  void ProcessFixups(u32 fixupOffset);

  float m_light_scale;

  Aabb m_aabb;
  Buffer m_rnd_seeds;
  Buffer m_accum_buffer;

  float m_scene_epsilon;
  int m_frame;

  vector<protocol::MeshBlob*> meshes;
  vector<protocol::NullObjectBlob*> nullObjects;
  vector<protocol::CameraBlob*> cameras;
  vector<protocol::LightBlob*> lights;
  vector<protocol::MaterialBlob*> materials;
  vector<protocol::SplineBlob*> splines;
  vector<protocol::BlobBase*> blobsById;
  vector<char> buf;
};

//------------------------------------------------------------------------------
bool MeshViewer::Load(const char* filename)
{
  // Low level loading - loads the mesh blob, fixes up pointers, and collects the
  // various blob types in various vectors.

  HANDLE h =
      CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (!h)
    return false;

  DWORD size = GetFileSize(h, NULL);
  if (size > 0)
  {
    buf.resize(size);
    DWORD res;
    if (!ReadFile(h, buf.data(), size, &res, NULL))
    {
      CloseHandle(h);
      return false;
    }
  }
  CloseHandle(h);

  // Process the blob - fixup pointers, and collects the various objects
  const protocol::SceneBlob* scene = (const protocol::SceneBlob*)&buf[0];

  if (strncmp(scene->id, "boba", 4) != 0)
    return false;

  ProcessFixups(scene->fixupOffset);

  auto fnAddBlob = [this](protocol::BlobBase* blob)
  {
    if (blob->id >= blobsById.size())
      blobsById.resize(blob->id + 1);
    blobsById[blob->id] = blob;
  };

  // null objects
  protocol::NullObjectBlob* nullBlob = (protocol::NullObjectBlob*)&buf[scene->nullObjectDataStart];
  for (u32 i = 0; i < scene->numNullObjects; ++i, ++nullBlob)
  {
    fnAddBlob(nullBlob);
    nullObjects.push_back(nullBlob);
  }

  // add meshes
  protocol::MeshBlob** meshBlobs = (protocol::MeshBlob**)&buf[scene->meshDataStart];
  for (u32 i = 0; i < scene->numMeshes; ++i, ++meshBlobs)
  {
    protocol::MeshBlob* meshBlob = *meshBlobs;
    // HACK(magnus): skip main cube for now
    //if (strcmp(meshBlob->name, "Cube") != 0)
    {
      fnAddBlob(meshBlob);
      meshes.push_back(meshBlob);
    }
  }

  // add lights
  protocol::LightBlob* lightBlob = (protocol::LightBlob*)&buf[scene->lightDataStart];
  for (u32 i = 0; i < scene->numLights; ++i, ++lightBlob)
  {
    fnAddBlob(lightBlob);
    lights.push_back(lightBlob);
  }

  // add cameras
  protocol::CameraBlob* cameraBlob = (protocol::CameraBlob*)&buf[scene->cameraDataStart];
  for (u32 i = 0; i < scene->numCameras; ++i, ++cameraBlob)
  {
    fnAddBlob(cameraBlob);
    cameras.push_back(cameraBlob);
  }

  // add materials
  char* ptr = &buf[scene->materialDataStart];
  for (u32 i = 0; i < scene->numMaterials; ++i)
  {
    protocol::MaterialBlob* materialBlob = (protocol::MaterialBlob*)ptr;
    materials.push_back(materialBlob);
    ptr += materialBlob->blobSize;
  }

  return true;
}

//------------------------------------------------------------------------------
void MeshViewer::ProcessFixups(u32 fixupOffset)
{
  // Process all the fixups. A list of locations that point to relative
  // data is stored (the fixup list), and for each of these locations, we
  // add the base of the file we loaded, converting the fixups to valid
  // memory locations

  // Note, on 64-bit, we are still limited to 32 bit file sizes and offsets, but
  // all the fixed up pointers are 64-bit.
  u32* fixupList = (u32*)&buf[fixupOffset];
  u32 numFixups = *fixupList++;
  intptr_t base = (intptr_t)&buf[0];
  u32* base32 = (u32*)base;
  for (u32 i = 0; i < numFixups; ++i)
  {
    // get the offset in the file that needs to be adjusted
    u32 src = fixupList[i];

    // adjust the offset from being a relativer pointer into the file
    // to being an absolute ptr into memory
    *(intptr_t*)(base + src) += base;
  }
}

//------------------------------------------------------------------------------
struct BobaOptixLoader
{
  optix::Aabb _aabb;
  optix::GeometryGroup _geometryGroup;

  optix::Matrix4x4 xformMatrixFromBlob(const protocol::BlobBase* blob)
  {
    float tx = blob->xformGlobal.pos[0];
    float ty = blob->xformGlobal.pos[1];
    float tz = blob->xformGlobal.pos[2];

    float rx = blob->xformGlobal.rot[0];
    float ry = blob->xformGlobal.rot[1];
    float rz = blob->xformGlobal.rot[2];

    float sx = blob->xformGlobal.scale[0];
    float sy = blob->xformGlobal.scale[1];
    float sz = blob->xformGlobal.scale[2];

    return
      optix::Matrix4x4::rotate(rx, make_float3(1, 0, 0)) *
      optix::Matrix4x4::rotate(ry, make_float3(0, 1, 0)) *
      optix::Matrix4x4::rotate(rz, make_float3(0, 0, 1)) *
      optix::Matrix4x4::scale(make_float3(sx, sy, sz)) *
      optix::Matrix4x4::translate(make_float3(tx, ty, tz));
  }

  void Load(const MeshViewer& loader, optix::Context& context, const string& mtlPath, const string& prgPath)
  {
    _geometryGroup = context->createGeometryGroup();
    unordered_map<int, optix::Material> materials;
    unordered_map<int, protocol::MaterialBlob*> materialBlobsById;

    Program prgClosestHit = context->createProgramFromPTXFile(mtlPath, "closest_hit_radiance");
    Program prgAnyHit = context->createProgramFromPTXFile(mtlPath, "any_hit_shadow");

    Program prgMeshIntersect = context->createProgramFromPTXFile(prgPath, "mesh_intersect");
    Program prgMeshBounds = context->createProgramFromPTXFile(prgPath, "mesh_bounds");

    for (protocol::MaterialBlob* materialBlob : loader.materials)
    {
      optix::Material mtl = context->createMaterial();
      mtl->setClosestHitProgram(RT_Radiance, prgClosestHit);
      mtl->setAnyHitProgram(RT_Shadow, prgAnyHit);
      materials[materialBlob->materialId] = mtl;
      materialBlobsById[materialBlob->materialId] = materialBlob;
    }

    for (protocol::MeshBlob* meshBlob : loader.meshes)
    {
      protocol::MeshBlob::DataStream* indexStream = nullptr;
      protocol::MeshBlob::DataStream* posStream = nullptr;
      protocol::MeshBlob::DataStream* normalStream = nullptr;
      protocol::MeshBlob::DataStream* uvStream = nullptr;

      int numTriangles = 0;
      int numVertices = 0;

      // collect streams
      for (int i = 0; i < meshBlob->streams->numElems; ++i)
      {
        protocol::MeshBlob::DataStream* ds = &meshBlob->streams->elems[i];
        if (strcmp(ds->name, "index32") == 0)
        {
          numTriangles = ds->dataSize / (3 * sizeof(int));
          int* tris = (int*)ds->data;
          for (int j = 0; j < numTriangles; ++j)
          {
            swap(tris[j*3+0], tris[j*3+2]);
          }
          indexStream = ds;
        }
        else if (strcmp(ds->name, "pos") == 0)
        {
          numVertices = ds->dataSize / (3 * sizeof(float));
          posStream = ds;
        }
        else if (strcmp(ds->name, "normal") == 0)
        {
          normalStream = ds;
        }
        else if (strcmp(ds->name, "uv") == 0)
        {
          uvStream = ds;
        }
      }

      // Create a optix::Geometry per material group
      for (int i = 0; i < meshBlob->materialGroups->numElems; ++i)
      {
        protocol::MeshBlob::MaterialGroup* group = &meshBlob->materialGroups->elems[i];

        optix::Material mat = materials[group->materialId];

        optix::Geometry geo = context->createGeometry();
        geo->setPrimitiveCount(numTriangles);
        geo->setIntersectionProgram(prgMeshIntersect);
        geo->setBoundingBoxProgram(prgMeshBounds);

        // Create buffers for available streams. Also copy over vertex data, and set them on the geo-object
        optix::Buffer vIndexBuffer, nIndexBuffer, tIndexBuffer;
        optix::Buffer vBuffer, nBuffer, tBuffer;

        const auto& copyStreamToBuffer = [](const protocol::MeshBlob::DataStream* ds, optix::Buffer& buffer) {
          void* ptr = buffer->map();
          memcpy(ptr, ds->data, ds->dataSize);
          buffer->unmap();
        };

        if (posStream)
        {
          vIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, numTriangles);
          geo["vindex_buffer"]->setBuffer(vIndexBuffer);
          copyStreamToBuffer(indexStream, vIndexBuffer);

          // Copy/transform vertices
          vBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
          geo["vertex_buffer"]->setBuffer(vBuffer);
          float3* src = (float3*)posStream->data;
          float3* dst = (float3*)vBuffer->map();

          optix::Matrix4x4 mtx = xformMatrixFromBlob(meshBlob);

          for (int i = 0; i < numVertices; ++i)
          {
            dst[i] = make_float3(mtx * make_float4(src[i], 1.0f));
            _aabb.include(dst[i]);
          }

          vBuffer->unmap();
        }

        if (normalStream)
        {
          nIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, numTriangles);
          geo["nindex_buffer"]->setBuffer(nIndexBuffer);
          copyStreamToBuffer(indexStream, nIndexBuffer);

          // Transform/copy normals.
          nBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVertices);
          geo["normal_buffer"]->setBuffer(nBuffer);
          float3* src = (float3*)normalStream->data;
          float3* dst = (float3*)nBuffer->map();

          optix::Matrix4x4 mtx = xformMatrixFromBlob(meshBlob);
          mtx = mtx.inverse().transpose();
          for (int i = 0; i < numVertices; ++i)
          {
            dst[i] = make_float3(mtx * make_float4(src[i], 0.0f));
          }

          nBuffer->unmap();
        }

        if (uvStream)
        {
          tIndexBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3, numTriangles);
          geo["tindex_buffer"]->setBuffer(tIndexBuffer);
          copyStreamToBuffer(indexStream, tIndexBuffer);

          tBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, numVertices);
          geo["texcoord_buffer"]->setBuffer(tBuffer);
          copyStreamToBuffer(uvStream, tBuffer);
        }

        {
          // per face material index. looked up in the triangle intersection
          optix::Buffer mbuffer =
              context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numTriangles);
          void* buf = mbuffer->map();
          memset(buf, 0, numTriangles * sizeof(int));
          mbuffer->unmap();

          geo["material_buffer"]->setBuffer(mbuffer);
        }

        GeometryInstance instance = context->createGeometryInstance(geo, &mat, &mat + 1);

        float3 zero = { 0, 0, 0 };
        float3 diffuse = zero;
        float3 ambient = zero;
        float3 specular = zero;
        float3 emissive = zero;

        if (group->materialId != ~0)
        {
          protocol::MaterialBlob* materialBlob = materialBlobsById[group->materialId];
          for (int i = 0; i < materialBlob->components->numElems; ++i)
          {
            protocol::MaterialBlob::MaterialComponent* c = &materialBlob->components->elems[i];
            if (strcmp(c->name, "color") == 0)
            {
              diffuse = make_float3(c->r, c->g, c->b);
            }
            else if (strcmp(c->name, "lumi") == 0)
            {
              float s = 1;
              emissive = make_float3(s * c->r, s * c->g, s * c->b);
            }
          }
        }

        instance["ambient_map"]->setTextureSampler(loadTexture(context, "", ambient));
        instance["diffuse_map"]->setTextureSampler(loadTexture(context, "", diffuse));
        instance["specular_map"]->setTextureSampler(loadTexture(context, "", specular));
        instance["emissive_map"]->setTextureSampler(loadTexture(context, "", emissive));

        _geometryGroup->addChild(instance);
      }

      optix::Acceleration acceleration = _geometryGroup->getAcceleration();
      AccelDescriptor accelDesc;

      if (!acceleration)
      {
        acceleration = context->createAcceleration(accelDesc.builder.c_str(), accelDesc.traverser.c_str());
        acceleration->setProperty("refine", accelDesc.refine);
        acceleration->setProperty("refit", accelDesc.refit);
        acceleration->setProperty("vertex_buffer_name",
            "vertex_buffer"); // Set these regardless of builder type. Ignored by some builders.
        acceleration->setProperty("index_buffer_name", "vindex_buffer");
        if (accelDesc.large_mesh)
          acceleration->setProperty("leaf_size", "1");
        _geometryGroup->setAcceleration(acceleration);
      }

      acceleration->markDirty();
    }
  }
};

//------------------------------------------------------------------------------
MeshViewer::MeshViewer()
    : m_light_scale(1.0f)
    , m_scene_epsilon(1e-4f)
    , m_frame(0)
{
}

//------------------------------------------------------------------------------
void MeshViewer::initScene(InitialCameraData& camera_data)
{
  initContext();
  initLights();

  genRndSeeds(getImageWidth(), getImageHeight());

  initGeometry();
  initCamera(camera_data);
  preprocess();
}

//------------------------------------------------------------------------------
void MeshViewer::initContext()
{
  m_context->setRayTypeCount(RT_COUNT);
  m_context->setEntryPointCount(1);
  m_context->setStackSize(4096);

  m_context["radiance_ray_type"]->setUint(RT_Radiance);
  m_context["shadow_ray_type"]->setUint(RT_Shadow);

  const unsigned int width = getImageWidth();
  const unsigned int height = getImageHeight();
  m_context["output_buffer"]->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, width, height));

  // Ray generation program setup
  const std::string camera_name = "pinhole_camera";
  const std::string camera_file = "accum_camera.cu";

  // The raygen program needs accum_buffer
  m_accum_buffer = m_context->createBuffer(
    RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4, width, height);
  m_context["accum_buffer"]->set(m_accum_buffer);
  resetAccumulation();

  const std::string camera_ptx = ptxpath("sample6", camera_file);
  Program ray_gen_program = m_context->createProgramFromPTXFile(camera_ptx, camera_name);
  m_context->setRayGenerationProgram(0, ray_gen_program);

  // Exception program
  const std::string except_ptx = ptxpath("sample6", camera_file);
  m_context->setExceptionProgram(0, m_context->createProgramFromPTXFile(except_ptx, "exception"));
  m_context["bad_color"]->setFloat(0.0f, 1.0f, 0.0f);

  // Miss program
  const std::string miss_ptx = ptxpath("sample6", "constantbg.cu");
  m_context->setMissProgram(0, m_context->createProgramFromPTXFile(miss_ptx, "miss"));
  m_context["bg_color"]->setFloat(0.34f, 0.55f, 0.85f);
}

//------------------------------------------------------------------------------
void MeshViewer::initLights()
{
  // Lights buffer
  BasicLight lights[] = {
      {make_float3(-60.0f, 30.0f, -120.0f), make_float3(0.2f, 0.2f, 0.25f) * m_light_scale, 0, 0},
      {make_float3(-60.0f, 0.0f, 120.0f), make_float3(0.1f, 0.1f, 0.10f) * m_light_scale, 0, 0},
      {make_float3(60.0f, 60.0f, 60.0f), make_float3(0.7f, 0.7f, 0.65f) * m_light_scale, 1, 0}};

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof(BasicLight));
  light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context["lights"]->set(light_buffer);
}

//------------------------------------------------------------------------------
void MeshViewer::initGeometry()
{
  double start, end;
  sutilCurrentTime(&start);

  // Load model
  Load(m_filename.c_str());

  BobaOptixLoader bobaLoader;
  bobaLoader.Load(*this,
      m_context,
      ptxpath("sample6", "one_bounce_diffuse.cu"),
      ptxpath("sample6", "triangle_mesh.cu"));

  m_aabb = bobaLoader._aabb;
  m_geometry_group = bobaLoader._geometryGroup;

  m_context["top_object"]->set(m_geometry_group);
  m_context["top_shadower"]->set(m_geometry_group);

  sutilCurrentTime(&end);
  std::cerr << "Time to load " << (m_accel_desc.large_mesh ? "and cluster " : "")
            << "geometry: " << end - start << " s.\n";
}

//------------------------------------------------------------------------------
void MeshViewer::initCamera(InitialCameraData& camera_data)
{
  // Set up camera
  float max_dim = m_aabb.maxExtent();
  float3 eye = m_aabb.center();
  eye.z += 2.0f * max_dim;
  eye.y -= 0.5f * max_dim;

  camera_data = InitialCameraData(eye, // eye
      m_aabb.center(),                 // lookat
      make_float3(0.0f, 1.0f, 0.0f),   // up
      30.0f);                          // vfov

  if (cameras.size() > 0)
  {
    protocol::CameraBlob* camera = cameras.front();
    protocol::BlobBase* target = blobsById[camera->targetId];

    camera_data = InitialCameraData(
      make_float3(camera->xformGlobal.pos[0], camera->xformGlobal.pos[1], camera->xformGlobal.pos[2]),
      make_float3(target->xformGlobal.pos[0], target->xformGlobal.pos[1], target->xformGlobal.pos[2]),
        make_float3(0.0f, 1.0f, 0.0f), // up
        30.0f);                        // vfov
  }

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context["eye"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
  m_context["U"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
  m_context["V"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
  m_context["W"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
}

//------------------------------------------------------------------------------
void MeshViewer::preprocess()
{
  // Settings which rely on previous initialization
  m_scene_epsilon = 1.e-4f * m_aabb.maxExtent();
  m_context["scene_epsilon"]->setFloat(m_scene_epsilon);

  // Prepare to run
  try
  {
    m_context->validate();
  }
  catch (exception e)
  {
    int a = 10;
  }
  double start, end_compile, end_AS_build;
  sutilCurrentTime(&start);
  m_context->compile();
  sutilCurrentTime(&end_compile);
  std::cerr << "Time to compile kernel: " << end_compile - start << " s.\n";
  m_context->launch(0, 0);
  sutilCurrentTime(&end_AS_build);
  std::cerr << "Time to build AS      : " << end_AS_build - end_compile << " s.\n";

  // Save cache file
  saveAccelCache();
}

//------------------------------------------------------------------------------
bool MeshViewer::keyPressed(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 'e':
      m_scene_epsilon *= .1f;
      std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
      m_context["scene_epsilon"]->setFloat(m_scene_epsilon);
      return true;
    case 'E':
      m_scene_epsilon *= 10.0f;
      std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
      m_context["scene_epsilon"]->setFloat(m_scene_epsilon);
      return true;
  }
  return false;
}

//------------------------------------------------------------------------------
void MeshViewer::doResize(unsigned int width, unsigned int height)
{
  // output_buffer resizing handled in base class
  m_accum_buffer->setSize(width, height);
  m_rnd_seeds->setSize(width, height);
  genRndSeeds(width, height);
  resetAccumulation();
}

//------------------------------------------------------------------------------
void MeshViewer::trace(const RayGenCameraData& camera_data)
{
  m_context["eye"]->setFloat(camera_data.eye);
  m_context["U"]->setFloat(camera_data.U);
  m_context["V"]->setFloat(camera_data.V);
  m_context["W"]->setFloat(camera_data.W);

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize(buffer_width, buffer_height);

  m_context->launch(0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height));
}

//------------------------------------------------------------------------------
void MeshViewer::cleanUp()
{
  SampleScene::cleanUp();
}

//------------------------------------------------------------------------------
Buffer MeshViewer::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}

//------------------------------------------------------------------------------
void MeshViewer::resetAccumulation()
{
  m_frame = 0;
  m_context["frame"]->setInt(m_frame);
  m_context["max_bounces"]->setInt(1);
  m_context["sqrt_diffuse_samples"]->setInt(8);
}

//------------------------------------------------------------------------------
void MeshViewer::genRndSeeds(unsigned int width, unsigned int height)
{
  // Init random number buffer if necessary.
  if (m_rnd_seeds.get() == 0)
  {
    m_rnd_seeds = m_context->createBuffer(
        RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, width, height);
    m_context["rnd_seeds"]->setBuffer(m_rnd_seeds);
  }

  unsigned int* seeds = static_cast<unsigned int*>(m_rnd_seeds->map());
  fillRandBuffer(seeds, width * height);
  m_rnd_seeds->unmap();
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  GLUTDisplay::init(argc, argv);

  GLUTDisplay::contDraw_E draw_mode = GLUTDisplay::CDNone;
  MeshViewer scene;

  // production quality :)
  string meshName = "d:/onedrive/tokko/gfx/sh_test1.boba";
  FILE* f = fopen(meshName.c_str(), "rb");
  if (f)
    fclose(f);
  else
    meshName[0] = 'c';
  scene.setMesh(meshName.c_str());

  try
  {
    GLUTDisplay::run("MeshViewer", &scene, draw_mode);
  }
  catch (Exception& e)
  {
    sutilReportError(e.getErrorString().c_str());
    exit(1);
  }

  return 0;
}
