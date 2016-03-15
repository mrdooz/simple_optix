
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

//-----------------------------------------------------------------------------
//
//  sample6.cpp: Renders a MeshScene
//  
//-----------------------------------------------------------------------------

#include <sutil.h>
#include <GLUTDisplay.h>
//#include <OptiXMesh.h>
#include "commonStructs.h"
#include "random.h"
#include "MeshScene.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <ImageLoader.h>

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

using namespace optix;

#include <stdint.h>
#include <vector>
#include <windows.h>
using namespace std;
typedef uint32_t u32;

namespace protocol
{
#ifndef BOBA_PROTOCOL_VERSION
#define BOBA_PROTOCOL_VERSION 5
#endif

#pragma pack(push, 1)

  enum
  {
    INVALID_OBJECT_ID = 0xffffffff
  };

  enum class LightType : u32
  {
    Point,
    Directional,
    Spot,
  };

  enum class FalloffType : u32
  {
    None = 0,
    Linear,
  };

  struct SceneBlob
  {
    char id[4];
    u32 version = 2;
    u32 flags;
    // offset of the deferred data
    u32 fixupOffset;
    u32 nullObjectDataStart;
    u32 meshDataStart;
    u32 lightDataStart;
    u32 cameraDataStart;
    u32 materialDataStart;
    u32 numNullObjects;
    u32 numMeshes;
    u32 numLights;
    u32 numCameras;
    u32 numMaterials;
#if BOBA_PROTOCOL_VERSION >= 2
    u32 splineDataStart;
    u32 numSplines;
#endif
  };

  struct Transform
  {
    float pos[3];
    float rot[3];
    float scale[3];
  };

  struct BlobBase
  {
    const char* name;
    u32 id;
    u32 parentId;
#if BOBA_PROTOCOL_VERSION < 4
    float mtxLocal[12];
    float mtxGlobal[12];
#endif
#if BOBA_PROTOCOL_VERSION >= 4
    Transform xformLocal;
    Transform xformGlobal;
#endif
  };

  struct MeshBlob : public BlobBase
  {
    struct DataStream
    {
      const char* name;
      u32 flags;
      u32 dataSize;
      void* data;
    };

    struct DataStreamArray
    {
      int numElems;
      DataStream* elems;
    };

    struct MaterialGroup
    {
      u32 materialId;
      u32 startIndex;
      u32 numIndices;
    };

    struct MaterialGroupArray
    {
      int numElems;
      MaterialGroup* elems;
    };

    // bounding sphere
    float sx, sy, sz, r;

    MaterialGroupArray* materialGroups;
    DataStreamArray* streams;
  };

  struct NullObjectBlob : public BlobBase
  {

  };

  struct CameraBlob : public BlobBase
  {
    float verticalFov;
    float nearPlane, farPlane;
#if BOBA_PROTOCOL_VERSION >= 3
    // new in version 3
    u32 targetId = INVALID_OBJECT_ID;
#endif
  };

  struct LightBlob : public BlobBase
  {
    LightType type;
    float color[4];
    float intensity;

    FalloffType falloffType;
    float falloffRadius;

    float outerAngle;
  };

  struct SplineBlob : public BlobBase
  {
    int type;
    u32 numPoints;
    float* points;
    bool isCLosed;
  };

  struct MaterialBlob
  {
    struct MaterialComponent
    {
      const char* name;
      float r, g, b, a;
      const char* texture;
      float brightness;
    };

    struct MaterialComponentArray
    {
      int numElems;
      MaterialComponent* elems;
    };

    u32 blobSize;
    const char* name;
    u32 materialId;
    MaterialComponentArray* components;
  };
#pragma pack(pop)
}

struct SceneLoader
{
  bool Load(const char* filename);
  void ProcessFixups(u32 fixupOffset);

  vector<protocol::MeshBlob*> meshes;
  vector<protocol::NullObjectBlob*> nullObjects;
  vector<protocol::CameraBlob*> cameras;
  vector<protocol::LightBlob*> lights;
  vector<protocol::MaterialBlob*> materials;
  vector<protocol::SplineBlob*> splines;
  vector<char> buf;
};

//------------------------------------------------------------------------------
bool SceneLoader::Load(const char* filename)
{
  HANDLE h = CreateFileA(
    filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
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

  const protocol::SceneBlob* scene = (const protocol::SceneBlob*)&buf[0];

  if (strncmp(scene->id, "boba", 4) != 0)
    return false;

  ProcessFixups(scene->fixupOffset);

  // null objects
  protocol::NullObjectBlob* nullBlob = (protocol::NullObjectBlob*)&buf[scene->nullObjectDataStart];
  for (u32 i = 0; i < scene->numNullObjects; ++i, ++nullBlob)
  {
    nullObjects.push_back(nullBlob);
  }

  // add meshes
  protocol::MeshBlob** meshBlob = (protocol::MeshBlob**)&buf[scene->meshDataStart];
  for (u32 i = 0; i < scene->numMeshes; ++i, ++meshBlob)
  {
    meshes.push_back(*meshBlob);
  }

  // add lights
  protocol::LightBlob* lightBlob = (protocol::LightBlob*)&buf[scene->lightDataStart];
  for (u32 i = 0; i < scene->numLights; ++i, ++lightBlob)
  {
    lights.push_back(lightBlob);
  }

  // add cameras
  protocol::CameraBlob* cameraBlob = (protocol::CameraBlob*)&buf[scene->cameraDataStart];
  for (u32 i = 0; i < scene->numCameras; ++i, ++cameraBlob)
  {
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
void SceneLoader::ProcessFixups(u32 fixupOffset)
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

  void Load(const SceneLoader& loader, optix::Context& context, const string& mtlPath, const string& prgPath)
  {
    _geometryGroup = context->createGeometryGroup();
    unordered_map<int, optix::Material> materials;

    Program prgClosestHit = context->createProgramFromPTXFile(mtlPath, "closest_hit_radiance");
    Program prgAnyHit = context->createProgramFromPTXFile(mtlPath, "any_hit_shadow");

    Program prgMeshIntersect = context->createProgramFromPTXFile(prgPath, "mesh_intersect");
    Program prgMeshBounds = context->createProgramFromPTXFile(prgPath, "mesh_bounds");

    for (protocol::MaterialBlob* materialBlob : loader.materials)
    {
      // TODO(magnus): enum for ray types
      optix::Material mtl = context->createMaterial();
      mtl->setClosestHitProgram(0u, prgClosestHit);
      mtl->setAnyHitProgram(1u, prgAnyHit);

      materials[materialBlob->materialId] = mtl;

      if (materialBlob->materialId != ~0)
      {
        for (int i = 0; i < materialBlob->components->numElems; ++i)
        {
          protocol::MaterialBlob::MaterialComponent* c = &materialBlob->components->elems[i];
          // TODO(magnus): set properties based on components
        }
      }
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
        //optix::Material mat = materials[group->materialId];
        optix::Material mat = materials[~0];

        optix::Geometry geo = context->createGeometry();
        geo->setPrimitiveCount(numTriangles);
        geo->setIntersectionProgram(prgMeshIntersect);
        geo->setBoundingBoxProgram(prgMeshBounds);

        // Create buffers for available streams. Also copy over vertex data, and set them on the geo-object
        optix::Buffer vIndexBuffer, nIndexBuffer, tIndexBuffer;
        optix::Buffer vBuffer, nBuffer, tBuffer;

        const auto& copyStreamToBuffer = [](const protocol::MeshBlob::DataStream* ds, optix::Buffer& buffer)
        {
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

          optix::Matrix4x4 mtx = optix::Matrix4x4::translate(*(float3*)&meshBlob->xformGlobal.pos[0]);

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

          optix::Matrix4x4 mtx = optix::Matrix4x4::translate(*(float3*)&meshBlob->xformGlobal.pos[0]);
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
          optix::Buffer mbuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numTriangles);
          void* buf = mbuffer->map();
          memset(buf, 0, numTriangles * sizeof(int));
          mbuffer->unmap();

          geo["material_buffer"]->setBuffer(mbuffer);
        }

        GeometryInstance instance = context->createGeometryInstance(geo, &mat, &mat + 1);

        // TODO(magnus): apply proper material settings
        float zero[3] = {0, 0, 0};
        float3 Kd = *(float3*)zero;
        float3 Ka = *(float3*)zero;
        float3 Ks = *(float3*)zero;

        instance["emissive"]->setFloat(0);
        instance["reflectivity"]->setFloat(0);
        instance["phong_exp"]->setFloat(0);
        instance["illum"]->setInt(0);

        instance["ambient_map"]->setTextureSampler(loadTexture(context, "", Ka));
        instance["diffuse_map"]->setTextureSampler(loadTexture(context, "", Kd));
        instance["specular_map"]->setTextureSampler(loadTexture(context, "", Ks));

        _geometryGroup->addChild(instance);
      }

      optix::Acceleration acceleration = _geometryGroup->getAcceleration();
      AccelDescriptor accelDesc;

      if (!acceleration)
      {
        acceleration = context->createAcceleration(accelDesc.builder.c_str(), accelDesc.traverser.c_str());
        acceleration->setProperty("refine", accelDesc.refine);
        acceleration->setProperty("refit", accelDesc.refit);
        acceleration->setProperty("vertex_buffer_name", "vertex_buffer"); // Set these regardless of builder type. Ignored by some builders.
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
//
// MeshViewer class 
//
//------------------------------------------------------------------------------
class MeshViewer : public MeshScene
{
public:
  //
  // Helper types
  //
  enum ShadeMode
  {
    SM_PHONG=0,
    SM_AO,
    SM_NORMAL,
    SM_ONE_BOUNCE_DIFFUSE,
    SM_AO_PHONG
  };

  enum CameraMode
  {
    CM_PINHOLE=0,
    CM_ORTHO
  };

  //
  // MeshViewer specific  
  //
  MeshViewer();

  // Setters for controlling application behavior
  void setShadeMode( ShadeMode mode )              { m_shade_mode = mode;               }
  void setCameraMode( CameraMode mode )            { m_camera_mode = mode;              }
  void setAORadius( float ao_radius )              { m_ao_radius = ao_radius;           }
  void setAOSampleMultiplier( int ao_sample_mult ) { m_ao_sample_mult = ao_sample_mult; }
  void setLightScale( float light_scale )          { m_light_scale = light_scale;       }
  void setAA( bool onoff )                         { m_aa_enabled = onoff;              }
  void setGroundPlane( bool onoff )                { m_ground_plane_enabled = onoff;    }
  void setAnimation( bool anim )                   { m_animation = anim;                }
  void setMergeMeshGroups( bool merge )            { m_merge_mesh_groups = merge;       }

  //
  // From SampleScene
  //
  virtual void   initScene( InitialCameraData& camera_data );
  virtual void   doResize( unsigned int width, unsigned int height );
  virtual void   trace( const RayGenCameraData& camera_data );
  virtual void   cleanUp();
  virtual bool   keyPressed(unsigned char key, int x, int y);
  virtual Buffer getOutputBuffer();

private:
  void initContext();
  void initLights();
  //void initMaterial();
  void initGeometry();
  void initCamera( InitialCameraData& cam_data );
  void preprocess();
  void resetAccumulation();
  void genRndSeeds( unsigned int width, unsigned int height );

  //void createGroundPlane(const OptiXMesh& optix_mesh);


  CameraMode    m_camera_mode;

  ShadeMode     m_shade_mode;
  bool          m_aa_enabled;
  bool          m_ground_plane_enabled;
  float         m_ao_radius;
  int           m_ao_sample_mult;
  float         m_light_scale;

  //Material      m_material;
  //MeshMaterialParams  m_material_params;
  Aabb          m_aabb;
  Buffer        m_rnd_seeds;
  Buffer        m_accum_buffer;
  bool          m_accum_enabled;

  float         m_scene_epsilon;
  int           m_frame;
  bool          m_animation;

  bool          m_merge_mesh_groups;
};


//------------------------------------------------------------------------------
//
// MeshViewer implementation
//
//------------------------------------------------------------------------------


MeshViewer::MeshViewer():
  m_camera_mode       ( CM_PINHOLE ),
  m_shade_mode        ( SM_PHONG ),
  m_aa_enabled        ( false ),
  m_ground_plane_enabled ( false ),
  m_ao_radius         ( 1.0f ),
  m_ao_sample_mult    ( 1 ),
  m_light_scale       ( 1.0f ),
  m_accum_enabled     ( false ),
  m_scene_epsilon     ( 1e-4f ),
  m_frame             ( 0 ),
  m_animation         ( false ),
  m_merge_mesh_groups ( true )
{
}


void MeshViewer::initScene( InitialCameraData& camera_data )
{
  initContext();
  initLights();
  //initMaterial();
  initGeometry();
  initCamera( camera_data );
  preprocess();
}


void MeshViewer::initContext()
{
  m_context->setRayTypeCount( 3 );
  m_context->setEntryPointCount( 1 );
  m_context->setStackSize( 1180 );

  m_context[ "radiance_ray_type"   ]->setUint( 0u );
  m_context[ "shadow_ray_type"     ]->setUint( 1u );
  m_context[ "max_depth"           ]->setInt( 5 );
  m_context[ "ambient_light_color" ]->setFloat( 0.2f, 0.2f, 0.2f );
  m_context[ "jitter_factor"       ]->setFloat( m_aa_enabled ? 1.0f : 0.0f );

  const unsigned int width = getImageWidth();
  const unsigned int height = getImageHeight();
  m_context[ "output_buffer"       ]->set( createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, width, height ) );
  
  m_accum_enabled = m_aa_enabled                         ||
                   m_shade_mode == SM_AO                 ||
                   m_shade_mode == SM_ONE_BOUNCE_DIFFUSE ||
                   m_shade_mode == SM_AO_PHONG;

  // Ray generation program setup
  const std::string camera_name = m_camera_mode == CM_PINHOLE ? "pinhole_camera" : "orthographic_camera"; 
  const std::string camera_file = m_accum_enabled             ? "accum_camera.cu" :
                                  m_camera_mode == CM_PINHOLE ? "pinhole_camera.cu"  :
                                                               "orthographic_camera.cu";

  if( m_accum_enabled ) {
    // The raygen program needs accum_buffer
    m_accum_buffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_FLOAT4,
                                            width, height );
    m_context["accum_buffer"]->set( m_accum_buffer );
    resetAccumulation();
  }

  const std::string camera_ptx  = ptxpath( "sample6", camera_file );
  Program ray_gen_program = m_context->createProgramFromPTXFile( camera_ptx, camera_name );
  m_context->setRayGenerationProgram( 0, ray_gen_program );


  // Exception program
  const std::string except_ptx  = ptxpath( "sample6", camera_file );
  m_context->setExceptionProgram( 0, m_context->createProgramFromPTXFile( except_ptx, "exception" ) );
  m_context[ "bad_color" ]->setFloat( 0.0f, 1.0f, 0.0f );


  // Miss program 
  const std::string miss_ptx = ptxpath( "sample6", "constantbg.cu" );
  m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( miss_ptx, "miss" ) );
  m_context[ "bg_color" ]->setFloat(  0.34f, 0.55f, 0.85f );
}


void MeshViewer::initLights()
{
  // Lights buffer
  BasicLight lights[] = {
    { make_float3( -60.0f,  30.0f, -120.0f ), make_float3( 0.2f, 0.2f, 0.25f )*m_light_scale, 0, 0 },
    { make_float3( -60.0f,   0.0f,  120.0f ), make_float3( 0.1f, 0.1f, 0.10f )*m_light_scale, 0, 0 },
    { make_float3(  60.0f,  60.0f,   60.0f ), make_float3( 0.7f, 0.7f, 0.65f )*m_light_scale, 1, 0 }
  };

  Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
  light_buffer->setFormat(RT_FORMAT_USER);
  light_buffer->setElementSize(sizeof( BasicLight ) );
  light_buffer->setSize( sizeof(lights)/sizeof(lights[0]) );
  memcpy(light_buffer->map(), lights, sizeof(lights));
  light_buffer->unmap();

  m_context[ "lights" ]->set( light_buffer );
}


//void MeshViewer::initMaterial()
//{
//  switch( m_shade_mode ) {
//    case SM_PHONG: {
//      // Use the default obj_material created by OptiXMesh if model has no material, but use this for the ground plane, if any
//      break;
//    }
//
//    case SM_NORMAL: {
//      const std::string ptx_path = ptxpath("sample6", "normal_shader.cu");
//      m_material = m_context->createMaterial();
//      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );
//      break;
//    }
//
//    case SM_AO: {
//      const std::string ptx_path = ptxpath("sample6", "ambocc.cu");
//      m_material = m_context->createMaterial();
//      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );
//      m_material->setAnyHitProgram    ( 1, m_context->createProgramFromPTXFile( ptx_path, "any_hit_occlusion" ) );    
//      break;
//    } 
//
//    case SM_ONE_BOUNCE_DIFFUSE: {
//      const std::string ptx_path = ptxpath("sample6", "one_bounce_diffuse.cu");
//      m_material = m_context->createMaterial();
//      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance" ) );
//      m_material->setAnyHitProgram    ( 1, m_context->createProgramFromPTXFile( ptx_path, "any_hit_shadow" ) );
//      break;
//    }
//
//    case SM_AO_PHONG: {
//      const std::string ptx_path = ptxpath("sample6", "ambocc.cu");
//      m_material = m_context->createMaterial();
//      m_material->setClosestHitProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "closest_hit_radiance_phong_ao" ) );
//      m_material->setAnyHitProgram    ( 1, m_context->createProgramFromPTXFile( ptx_path, "any_hit_shadow" ) );
//      m_material->setAnyHitProgram    ( 2, m_context->createProgramFromPTXFile( ptx_path, "any_hit_occlusion" ) );
//
//      // the ao phong shading uses monochrome single-float values for Kd, etc.,
//      // so it won't make sense to use the float4 colors from m_default_material_params
//      m_context["Kd"]->setFloat(1.0f);
//      m_context["Ka"]->setFloat(0.6f);
//      m_context["Ks"]->setFloat(0.0f);
//      m_context["Kr"]->setFloat(0.0f);
//      m_context["phong_exp"]->setFloat(0.0f);
//      break;
//    }
//  }
//
//  if( m_accum_enabled ) {
//    genRndSeeds( getImageWidth(), getImageHeight() );
//  }
//}


void MeshViewer::initGeometry()
{
  double start, end;
  sutilCurrentTime(&start);

  // Load model 
  SceneLoader sceneLoader;
  sceneLoader.Load(m_filename.c_str());

  BobaOptixLoader bobaLoader;
  bobaLoader.Load(sceneLoader, m_context,
                  ptxpath("sample6", "normal_shader.cu"),
                  ptxpath("sample6", "triangle_mesh.cu"));

  m_aabb = bobaLoader._aabb;
  m_geometry_group = bobaLoader._geometryGroup;

  m_context["top_object"]->set(m_geometry_group);
  m_context["top_shadower"]->set(m_geometry_group);


  //OptiXMesh loader( m_context, m_geometry_group, m_accel_desc );
  //loader.setMergeMeshGroups( m_merge_mesh_groups );
  //loader.loadBegin_Geometry( m_filename );

  //// Override default OptiXMesh material for most shade modes
  //if( m_shade_mode == SM_NORMAL || m_shade_mode == SM_AO || m_shade_mode == SM_AO_PHONG
  //    || m_shade_mode == SM_ONE_BOUNCE_DIFFUSE )
  //{
  //  for( size_t i = 0; i < loader.getMaterialCount(); ++i ) {
  //    loader.setOptiXMaterial( static_cast<int>(i), m_material );
  //  }
  //}
    
  //m_aabb = loader.getSceneBBox();
  //
  //// Add ground plane to loader so it will get the right materials and be added to the group.
  //if( m_ground_plane_enabled )
  //  createGroundPlane(loader);

  //loader.loadFinish_Materials();


  //// Load acceleration structure from a file if that was enabled on the
  //// command line, and if we can find a cache file. Note that the type of
  //// acceleration used will be overridden by what is found in the file.
  //loadAccelCache();

  //m_context[ "top_object" ]->set( m_geometry_group );
  //m_context[ "top_shadower" ]->set( m_geometry_group );

  sutilCurrentTime(&end);
  //std::cerr << "Triangles:" << loader.getNumTriangles() << std::endl;
  std::cerr << "Time to load " << (m_accel_desc.large_mesh ? "and cluster " : "") << "geometry: " << end-start << " s.\n";
}


//void MeshViewer::createGroundPlane(const OptiXMesh& optix_mesh)
//{
//  Geometry ground_geom = m_context->createGeometry();
//  ground_geom->setPrimitiveCount( 1u );
//  ground_geom->setBoundingBoxProgram( m_context->createProgramFromPTXFile( ptxpath( "sample6", "parallelogram.cu" ), "bounds" ) );
//  ground_geom->setIntersectionProgram( m_context->createProgramFromPTXFile( ptxpath( "sample6", "parallelogram.cu" ),"intersect" ) );
//
//  const float GROUND_SCALE = 5.0f;
//  float ex = m_aabb.maxExtent() * GROUND_SCALE;
//
//  float3 anchor = make_float3( -ex*0.5f, m_aabb.m_min.y, -ex*0.5f);
//  float3 v1 = make_float3( 0, 0, ex);
//  float3 v2 = make_float3( ex, 0, 0);
//  float3 normal = cross( v1, v2 );
//  normal = normalize( normal );
//  float d = dot( normal, anchor );
//  v1 *= 1.0f/dot( v1, v1 );
//  v2 *= 1.0f/dot( v2, v2 );
//  float4 plane = make_float4( normal, d );
//  ground_geom["plane"]->setFloat( plane );
//  ground_geom["v1"]->setFloat( v1 );
//  ground_geom["v2"]->setFloat( v2 );
//  ground_geom["anchor"]->setFloat( anchor );
//
//  GeometryInstance GI = m_context->createGeometryInstance();
//  GI->setGeometry( ground_geom );
//
//  if( m_shade_mode == SM_AO || m_shade_mode == SM_AO_PHONG || m_shade_mode == SM_NORMAL ) {
//    GI->addMaterial( m_material );
//  }
//  else {
//    optix_mesh.setOptixInstanceMatParams( GI, optix_mesh.getMeshMaterialParams(0) );
//    GI->addMaterial( optix_mesh.getOptiXMaterial( 0 ) );
//  }
//
//  m_geometry_group->addChild(GI);
//}


void MeshViewer::initCamera( InitialCameraData& camera_data )
{
  // Set up camera
  float max_dim  = m_aabb.maxExtent();
  float3 eye     = m_aabb.center();
  eye.z         += 2.0f * max_dim;
  eye.y -= 0.5f * max_dim;

  camera_data = InitialCameraData( eye,                             // eye
                                   m_aabb.center(),                  // lookat
                                   make_float3( 0.0f, 1.0f, 0.0f ), // up
                                   30.0f );                         // vfov

  // Declare camera variables.  The values do not matter, they will be overwritten in trace.
  m_context[ "eye"]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "U"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "V"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );
  m_context[ "W"  ]->setFloat( make_float3( 0.0f, 0.0f, 0.0f ) );

}


void MeshViewer::preprocess()
{
  // Settings which rely on previous initialization
  m_scene_epsilon = 1.e-4f * m_aabb.maxExtent();
  m_context[ "scene_epsilon"      ]->setFloat( m_scene_epsilon );
  m_context[ "occlusion_distance" ]->setFloat( m_aabb.maxExtent() * 0.3f * m_ao_radius );

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
  std::cerr << "Time to compile kernel: "<<end_compile-start<<" s.\n";
  m_context->launch(0,0);
  sutilCurrentTime(&end_AS_build);
  std::cerr << "Time to build AS      : "<<end_AS_build-end_compile<<" s.\n";

  // Save cache file
  saveAccelCache();
}


bool MeshViewer::keyPressed(unsigned char key, int x, int y)
{
   switch (key)
   {
     case 'e': m_scene_epsilon *= .1f;
       std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
       m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
       return true;
     case 'E':
       m_scene_epsilon *= 10.0f;
       std::cerr << "scene_epsilon: " << m_scene_epsilon << std::endl;
       m_context[ "scene_epsilon" ]->setFloat( m_scene_epsilon );
       return true;
   }
   return false;
}

          
void MeshViewer::doResize( unsigned int width, unsigned int height )
{
  // output_buffer resizing handled in base class
  if( m_accum_enabled ) {
    m_accum_buffer->setSize( width, height );
    m_rnd_seeds->setSize( width, height );
    genRndSeeds( width, height );
    resetAccumulation();
  }
}

void MeshViewer::trace( const RayGenCameraData& camera_data )
{
  m_context["eye"]->setFloat(camera_data.eye);
  m_context["U"]->setFloat(camera_data.U);
  m_context["V"]->setFloat(camera_data.V);
  m_context["W"]->setFloat(camera_data.W);

  Buffer buffer = m_context["output_buffer"]->getBuffer();
  RTsize buffer_width, buffer_height;
  buffer->getSize( buffer_width, buffer_height );

  m_context->launch( 0, static_cast<unsigned int>(buffer_width), static_cast<unsigned int>(buffer_height) );
}


void MeshViewer::cleanUp()
{
  SampleScene::cleanUp();
}


Buffer MeshViewer::getOutputBuffer()
{
  return m_context["output_buffer"]->getBuffer();
}


void MeshViewer::resetAccumulation()
{
  m_frame = 0;
  m_context[ "frame"                  ]->setInt( m_frame );
  m_context[ "sqrt_occlusion_samples" ]->setInt( 1 * m_ao_sample_mult );
  m_context[ "sqrt_diffuse_samples"   ]->setInt( 1 );
}


void MeshViewer::genRndSeeds( unsigned int width, unsigned int height )
{
  // Init random number buffer if necessary.
  if( m_rnd_seeds.get() == 0 ) {
    m_rnd_seeds = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT,
                                         width, height );
    m_context["rnd_seeds"]->setBuffer(m_rnd_seeds);
  }

  unsigned int* seeds = static_cast<unsigned int*>( m_rnd_seeds->map() );
  fillRandBuffer(seeds, width*height);
  m_rnd_seeds->unmap();
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0, bool doExit = true )
{
  std::cerr
    << "Usage  : " << argv0 << " [options]\n"
    << "App options:\n"
    << "  -h  | --help                               Print this usage message\n"
    << "  -o  | --obj <file>                         Specify model to be rendered (OBJ or PLY)\n"
    << "  -c  | --cache                              Turn on acceleration structure caching\n"
    << "  -a  | --ao-shade                           Use progressive ambient occlusion shader\n"
    << "  -ap | --ao-phong-shade                     Use progressive ambient occlusion and phong shader\n"
    << "  -aa | --antialias                          Use subpixel jittering to perform antialiasing\n"
    << "  -n  | --normal-shade                       Use normal shader\n"
    << "  -i  | --diffuse-shade                      Use one bounce diffuse shader\n"
    << "  -O  | --ortho                              Use orthographic camera (cannot use AO mode with ortho)\n"
    << "  -r  | --ao-radius <scale>                  Scale ambient occlusion radius\n"
    << "  -m  | --ao-sample-mult <n>                 Multiplier for the number of AO samples\n"
    << "  -l  | --light-scale <scale>                Scale lights by constant factor\n"
    << "  -g  | --ground                             Create a ground plane\n"
    << "        --animation                          Spin the model (useful for benchmarking)\n"
    << "        --merge-groups-off                   Turn off mesh group merging\n"
    << std::endl;
  GLUTDisplay::printUsage();

  std::cerr
    << "App keystrokes:\n"
    << "  e Decrease scene epsilon size (used for shadow ray offset)\n"
    << "  E Increase scene epsilon size (used for shadow ray offset)\n"
    << std::endl;

  if ( doExit ) exit(1);
}


int main( int argc, char** argv ) 
{
  GLUTDisplay::init( argc, argv );
  
  GLUTDisplay::contDraw_E draw_mode = GLUTDisplay::CDNone; 
  MeshViewer scene;
  scene.setMesh("d:/onedrive/tokko/gfx/sh_test1.boba");
  //scene.setMesh( (std::string( sutilSamplesDir() ) + "/simpleAnimation/cow.obj").c_str() );

  for ( int i = 1; i < argc; ++i ) {
    std::string arg( argv[i] );
    if( arg == "-n" || arg == "--normal-shade" ) {
      scene.setShadeMode( MeshViewer::SM_NORMAL );
    } else if( arg == "-a" || arg == "--ao-shade" ) {
      scene.setShadeMode( MeshViewer::SM_AO);
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-i" || arg == "--diffuse-shade" ) {
      scene.setShadeMode( MeshViewer::SM_ONE_BOUNCE_DIFFUSE );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-ap" || arg == "--ao-phong-shade" ) {
      scene.setShadeMode( MeshViewer::SM_AO_PHONG );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-aa" || arg == "--antialias" ) {
      scene.setAA( true );
      draw_mode = GLUTDisplay::CDProgressive;
    } else if( arg == "-g" || arg == "--ground" ) {
      scene.setGroundPlane( true );
    } else if( arg == "-O" || arg == "--ortho" ) {
      scene.setCameraMode( MeshViewer::CM_ORTHO );
    } else if( arg == "-h" || arg == "--help" ) {
      printUsageAndExit( argv[0] ); 
    } else if( arg == "-o" || arg == "--obj" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setMesh( argv[++i] );
    } else if( arg == "--animation" ) {
      scene.setAnimation( true );
    } else if( arg == "-r" || arg == "--ao-radius" ) {
      if ( i == argc-1 ) printUsageAndExit( argv[0] );
      scene.setAORadius( static_cast<float>( atof( argv[++i] ) ) );
    } else if( arg == "-m" || arg == "--ao-sample-mult" ) {
      if(i == argc - 1) printUsageAndExit(argv[0]);
      scene.setAOSampleMultiplier(atoi(argv[++i]));
    } else if( arg == "--merge-groups-off" ) {
      scene.setMergeMeshGroups( false );
    } else if( arg == "-l" || arg == "--light-scale" ) {
      if(i == argc - 1) printUsageAndExit(argv[0]);
      scene.setLightScale(static_cast<float>(atof(argv[++i])));
    } else {
      std::cerr << "Unknown option: '" << arg << "'" << std::endl;
      printUsageAndExit( argv[0] );
    }
  }
  
  if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );

  try {
    GLUTDisplay::run( "MeshViewer", &scene, draw_mode );
  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(1);
  }

  return 0;
}
