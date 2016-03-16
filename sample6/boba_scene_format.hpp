#pragma once

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
    bool isClosed;
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
