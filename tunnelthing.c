/*
 * gcc -o missile missile.c -lm -lSDL2 -I/usr/include/SDL2 -lGLESv2
 */

#define _GNU_SOURCE

#define EGL_EGLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <SDL.h>
#include <SDL_opengles2.h>

#include <unistd.h>

#define CLIP 20.0f
#define FOV 1.7320508075688774f
#define DRIFT_DAMPING 0.25f

typedef enum MG3D_BARRIER_TYPE {
  MG3D_BARRIER_BLANK,
  MG3D_BARRIER_1,
  MG3D_BARRIER_2,
  MG3D_BARRIER_3,
  MG3D_BARRIER_4,
  MG3D_BARRIER_5,
  MG3D_BARRIER_6,
  MG3D_BARRIER_FINISH,
  MG3D_TOTAL_BARRIERS
} BarrierType;

typedef struct {
  GLuint program;
  GLint viewProjectionMatrix, modelMatrix,  modelZ, grey, pixSize;
} BarrierProgram;

typedef struct {
  float offset;
  float rotspeed;
  uint32_t base_index;
} DrawBarrier;

static DrawBarrier draw_barriers[] = {
  { 0.00f,  1.5f, 2 },
  { 0.25f,  2.1f, 6 },
  { 0.50f,  0.6f, 4 },
  { 0.75f, -0.9f, 5 }
};

static SDL_Window * window;
static int width, height;
static float vwidth, vheight, currentx, currenty;
static float fovaspect;

static BarrierProgram barrier_programs[MG3D_TOTAL_BARRIERS];
static GLuint barrier_vertex_vbo;
static GLuint barrier_index_vbo;

static GLuint rail_vertex_vbo;
static GLuint rail_index_vbo;
static GLuint rail_program;
static GLint rail_uniform_model_matrix;
static GLint rail_uniform_view_projection_matrix;

static GLfloat ViewProjectionMatrix[16] = {
  0.0f,  0.0f,  0.0f,  0.0f,
  0.0f,   FOV,  0.0f,  0.0f,
  0.0f,  0.0f, -1.0f, -1.0f,
  0.0f,  0.0f,  CLIP,  CLIP
};

static SDL_GLContext gl_context;
static SDL_bool rendering = SDL_TRUE;

#ifndef __GLIBC__
/* I love GNU libc. */
void sincosf(float x, float *_sin, float *_cos)
{
  *_sin = sinf(x);
  *_cos = cosf(x);
}
#endif

static inline GLfloat
clamp(GLfloat v, GLfloat min, GLfloat max)
{
  return v > max ? max : v < min ? min : v;
}

static inline void
polar(GLfloat x, GLfloat y, GLfloat *rho, GLfloat *theta)
{
  *rho  = sqrtf(x * x + y * y);
  *theta = atan2f(y, x);
}

static inline void
euclidean(GLfloat rho, GLfloat theta, GLfloat *x, GLfloat *y)
{
  GLfloat s, c;
  sincosf(theta, &s, &c);
  *x = rho * c;
  *y = rho * s;
}

static const GLfloat rail_vertices[] = {
 -0.0940f,  0.4910f,  0.0f,
 -0.0625f,  0.4960f,  0.0f,
 -0.0940f,  0.4910f,  CLIP,
 -0.0625f,  0.4960f,  CLIP,

  0.4375f,  0.2420f,  0.0f,
  0.4520f,  0.2125f,  0.0f,
  0.4375f,  0.2420f,  CLIP,
  0.4520f,  0.2125f,  CLIP,

  0.3645f, -0.3415f,  0.0f,
  0.3420f, -0.3640f,  0.0f,
  0.3645f, -0.3415f,  CLIP,
  0.3420f, -0.3640f,  CLIP,

 -0.2120f, -0.4520f,  0.0f,
 -0.2410f, -0.4375f,  0.0f,
 -0.2120f, -0.4520f,  CLIP,
 -0.2410f, -0.4375f,  CLIP,

 -0.4950f,  0.0625f,  0.0f,
 -0.4910f,  0.0935f,  0.0f,
 -0.4950f,  0.0625f,  CLIP,
 -0.4910f,  0.0935f,  CLIP
};

static const GLushort rail_indices[] = {
   0,  1,  2,
   1,  2,  3,
   4,  5,  6,
   5,  6,  7,
   8,  9, 10,
   9, 10, 11,
  12, 13, 14,
  13, 14, 15,
  16, 17, 18,
  17, 18, 19
};

static const char rail_vert_shader[] =
  "precision lowp float;"
  "attribute vec3 position;"

  "uniform mat4 viewProjectionMatrix;"
  "uniform mat2 modelMatrix;"

  "void main() {"
    "gl_Position = viewProjectionMatrix * vec4(modelMatrix * position.xy, position.z, 1.0);"
  "}";

static const char rail_frag_shader[] =
  "precision lowp float;"

  "void main() {"
    "float depth = gl_FragCoord.z / gl_FragCoord.w;"
    "gl_FragColor = vec4(vec3(1.0 - clamp(exp2(-0.04 * depth * depth * 1.442695), 0.0, 1.0)), 1.0);"
  "}";

static const GLfloat barrier_vertices[] = {
  0.5468f,  0.0000f,
  0.5052f,  0.2093f,
  0.3866f,  0.3866f,
  0.2093f,  0.5052f,
  0.0000f,  0.5468f,
 -0.2093f,  0.5052f,
 -0.3866f,  0.3866f,
 -0.5052f,  0.2093f,
 -0.5468f,  0.0000f,
 -0.5052f, -0.2093f,
 -0.3866f, -0.3866f,
 -0.2093f, -0.5052f,
  0.0000f, -0.5468f,
  0.2093f, -0.5052f,
  0.3866f, -0.3866f,
  0.5052f, -0.2093f
};

static const GLushort barrier_indices[] = {
   0,  1,  2,
   2,  3,  4,
   4,  5,  6,
   6,  7,  8,
   8,  9, 10,
  10, 11, 12,
  12, 13, 14,
  14, 15,  0,
   0,  2,  4,
   4,  6,  8,
   8, 10, 12,
  12, 14,  0,
   0,  4,  8,
   8, 12,  0
};

static const GLchar barrier_vert_shader[] =
  "precision lowp float;"
  "attribute vec2 position;"

  "uniform float pixSize;"
  "uniform float modelZ;"
  "uniform mat2 modelMatrix;"
  "uniform mat4 viewProjectionMatrix;"

  "varying vec2 euclid;"
  "varying float pixSizeFinal;"
  "varying float fade;"
  "varying float fogIntensity;"

  "void main() {"
    "euclid = position;"
    "gl_Position = viewProjectionMatrix * vec4(modelMatrix * position, modelZ, 1.0);"
    "pixSizeFinal = pixSize * gl_Position.z;"
    "fade = smoothstep(0.01, 0.2, gl_Position.z);"
    "fogIntensity = clamp(exp2(-0.04 * gl_Position.z * gl_Position.z * 1.442695), 0.0, 1.0);"
  "}";

static const GLchar barrier_base_frag[] =
  "precision lowp float;"

  "varying vec2 euclid;"
  "varying float pixSizeFinal;"
  "varying float fade;"
  "varying float fogIntensity;"
  "uniform float grey;"

  "float sdfLine(vec2 a, vec2 b, vec2 coord) {"
    "vec2 ld = b - a;"
    "vec2 pd = vec2(ld.y, -ld.x);"
    "vec2 dp = a - coord;"
    "return dot(normalize(pd), dp);"
  "}"

  "float sdfCircle( const vec2 c, const float r, const vec2 coord ) {"
    "vec2 offset = coord - c;"

    "return sqrt(offset.x * offset.x + offset.y * offset.y) - r;"
  "}"

  "float sdfUnion( const float a, const float b ) {"
    "return min(a, b);"
  "}"

  "float sdfUnion( const float a, const float b, const float c ) {"
    "return min(a, min(b, c));"
  "}"

  "float sdfUnion( const float a, const float b, const float c, const float d ) {"
    "return min(a, min(b, min(c, d)));"
  "}"

  "float sdfDifference( const float a, const float b) {"
    "return max(a, -b);"
  "}"

  "float sdfIntersection( const float a, const float b ) {"
    "return max(a, b);"
  "}"

  "vec2 render( const float distance, const float pixSizeFinal, const float strokeWidth, const float halfGamma, const float grey) {"
    "float halfStroke = strokeWidth * 0.5;"

    "if (distance < -halfStroke + pixSizeFinal) {"
      "float factor = smoothstep(-halfStroke - pixSizeFinal, -halfStroke + pixSizeFinal, distance);"
      "return vec2(mix(grey, 0.0, factor * halfGamma), fade);"
    "} else if (distance <= halfStroke - pixSizeFinal) {"
      "return vec2(0.0, fade);"
    "} else {"
      "float factor = smoothstep(halfStroke - pixSizeFinal, halfStroke + pixSizeFinal, distance);"
      "return vec2(0.0, (1.0 - factor * halfGamma) * fade);"
    "}"
  "}"

  "float signedDistance( const vec2 coord ) {";


static const GLchar *barrier_samplers[MG3D_TOTAL_BARRIERS] = {
  /* MG3D_BARRIER_BLANK */
  "return sdfDifference("
    "sdfCircle(vec2(0.0), 0.5, coord),"
    "sdfCircle(vec2(0.0), 0.45, coord));",

  /* MG3D_BARRIER_1 */
  "return sdfIntersection("
    "sdfCircle(vec2(0.0), 0.5, coord),"
    "sdfUnion("
      "sdfIntersection(-coord.y, -coord.x),"
      "sdfIntersection(coord.y, coord.x),"
      "sdfCircle(vec2(0.0), 0.15, coord),"
      "sdfDifference(sdfCircle(vec2(0.0), 0.5, coord), sdfCircle(vec2(0.0), 0.45, coord))));",

  /* MG3D_BARRIER_2 */
  "return sdfDifference("
    "sdfCircle(vec2(0.0), 0.5, coord),"
    "sdfUnion("
      "sdfCircle(vec2(0.250, 0.0), 0.15, coord),"
      "sdfCircle(vec2(-0.125, 0.2165), 0.15, coord),"
      "sdfCircle(vec2(-0.125, -0.2165), 0.15, coord)));",

  /* MG3D_BARRIER_3 */
  "return sdfIntersection("
    "sdfCircle(vec2(0.0), 0.5, coord),"
    "sdfUnion("
      "sdfDifference("
        "sdfUnion(coord.y, -coord.x),"
        "sdfCircle(vec2(0.0), 0.15, coord)),"
      "sdfDifference("
        "sdfCircle(vec2(0.0), 0.5, coord),"
        "sdfCircle(vec2(0.0), 0.45, coord))));",

  /* MG3D_BARRIER_4 */
  "return sdfUnion("
    "sdfCircle(vec2(0.0), 0.15, coord),"
    "sdfDifference("
      "sdfCircle(vec2(0.0), 0.5, coord),"
      "sdfIntersection("
        "sdfIntersection(sdfLine(vec2(0.0), vec2(0.154, 0.423), coord), coord.y),"
        "sdfCircle(vec2(0.0), 0.45, coord))));",

  /* MG3D_BARRIER_5 */
  "return sdfIntersection("
    "sdfCircle(vec2(0.0), 0.5, abs(coord)),"
    "sdfUnion("
      "sdfIntersection(-sdfCircle(vec2(0.40796, 0.0), 0.29, abs(coord)), abs(coord).x - 0.225),"
      "sdfDifference("
        "sdfCircle(vec2(0.0), 0.5, abs(coord)),"
        "sdfCircle(vec2(0.0), 0.45, abs(coord)))));",

  /* MG3D_BARRIER_6 */
  "return sdfDifference("
    "sdfCircle(vec2(0.0), 0.5, abs(coord)),"
    "sdfIntersection(abs(coord).x - 0.4, abs(coord).y - 0.125));",

  /* MG3D_BARRIER_FINISH */
  "return sdfIntersection("
    "sdfUnion("
      "sdfDifference(sdfCircle(vec2(0.0), 0.5, coord), sdfCircle(vec2(0.0), 0.45, coord)),"
      "sdfDifference("
        "sdfUnion("
          "sdfDifference(-sdfUnion(sdfLine(vec2(-0.318, 0.318), vec2(0.318, -0.318), coord), coord.x), -coord.y),"
          "sdfDifference(-sdfUnion(-sdfLine(vec2(-0.318, 0.318), vec2(0.318, -0.318), coord), -coord.x), coord.y)),"
        "sdfCircle(vec2(0.0), 0.375, coord))),"
    "sdfCircle(vec2(0.0), 0.5, coord));",
};

static const GLchar barrier_end_frag[] =
  "}"

  "void main() {"
    "float d = signedDistance(euclid);"
    "vec2 sample = render(d, pixSizeFinal, 0.010, 1.1, grey);"
    "gl_FragColor = vec4(vec3(mix(1.0, sample.x, fogIntensity) * sample.y), sample.y);"
  "}";

static inline void
rotationmat2 (GLfloat m[4], GLfloat angle)
{
  sincosf(angle, &m[1], &m[0]);
  m[2] = -m[1];
  m[3] =  m[0];
}

static void
reshape_viewport()
{
  BarrierType i;
  static int vwidthi, vheighti;
  static float pix_size;

  SDL_GetWindowSize(window, &vwidthi, &vheighti);
  vwidth = (float)vwidthi;
  vheight = (float)vheighti;

  SDL_GL_GetDrawableSize(window, &width, &height);

  pix_size = 1.0f / (float)(width < height? width: height);

  glViewport(0, 0, (GLint)width, (GLint)height);
  fovaspect = FOV / ((float)width / (float)height);
  ViewProjectionMatrix[0] = fovaspect;

  for(i = MG3D_BARRIER_BLANK; i < MG3D_TOTAL_BARRIERS; i++) {
    glUseProgram(barrier_programs[i].program);
    glUniform1f(barrier_programs[i].pixSize, pix_size);
  }
}

static inline void
handle_SDL_events()
{
  static SDL_Event event;
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_KEYDOWN:
      if (event.key.keysym.sym == SDLK_ESCAPE)
        rendering = SDL_FALSE;
      break;

    case SDL_QUIT:
      rendering = SDL_FALSE;
      break;

    case SDL_WINDOWEVENT:
      if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
        reshape_viewport();
      break;
    }
  }
}

static void
init()
{
  BarrierType i;
  GLuint vert, frag, program;
  const char *source;
  char msg[256];
  msg[0] = '\0';

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glEnableVertexAttribArray(0);
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

  source = barrier_vert_shader;
  vert = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vert, 1, &source, NULL);
  glCompileShader(vert);
  glGetShaderInfoLog(vert, (GLsizei)sizeof(msg), NULL, msg);
  if(msg[0] != '\0') SDL_Log("Barrier vertex shader compile info: %s\n", msg);

  for (i = MG3D_BARRIER_BLANK; i < MG3D_TOTAL_BARRIERS; i++) {
    frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 3, (const GLchar * []) {barrier_base_frag, barrier_samplers[i], barrier_end_frag}, NULL);
    glCompileShader(frag);
    glGetShaderInfoLog(frag, (GLsizei)sizeof(msg), NULL, msg);
    if(msg[0] != '\0') SDL_Log("Barrier %d fragment shader compile info: %s\n", i, msg);
    program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glGetProgramInfoLog(program, (GLsizei)sizeof(msg), NULL, msg);
    if(msg[0] != '\0') SDL_Log("Barrier %d fragment shader link info: %s\n", i, msg);
    glUseProgram(program);
    glBindAttribLocation(program, 0, "position");
    barrier_programs[i] = (BarrierProgram) {
      .program = program,
      .viewProjectionMatrix = glGetUniformLocation(program, "viewProjectionMatrix"),
      .modelMatrix = glGetUniformLocation(program, "modelMatrix"),
      .modelZ = glGetUniformLocation(program, "modelZ"),
      .grey = glGetUniformLocation(program, "grey"),
      .pixSize = glGetUniformLocation(program, "pixSize")
    };
  }

  source = rail_vert_shader;
  vert = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vert, 1, &source, NULL);
  glCompileShader(vert);
  glGetShaderInfoLog(vert, (GLsizei)sizeof(msg), NULL, msg);
  if(msg[0] != '\0') SDL_Log("rail vertex shader info: %s\n", msg);
  source = rail_frag_shader;
  frag = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(frag, 1, &source, NULL);
  glCompileShader(frag);
  glGetShaderInfoLog(frag, (GLsizei)sizeof(msg), NULL, msg);
  if(msg[0] != '\0') SDL_Log("rail fragment shader info: %s\n", msg);
  rail_program = glCreateProgram();
  glAttachShader(rail_program, vert);
  glAttachShader(rail_program, frag);
  glBindAttribLocation(rail_program, 0, "position");
  glLinkProgram(rail_program);
  glGetProgramInfoLog(rail_program, (GLsizei)sizeof(msg), NULL, msg);
  if(msg[0] != '\0') SDL_Log("info: %s\n", msg);
  glUseProgram(rail_program);
  rail_uniform_model_matrix = glGetUniformLocation(rail_program, "modelMatrix");
  rail_uniform_view_projection_matrix = glGetUniformLocation(rail_program, "viewProjectionMatrix");

  /* set up array buffers for rails */
  glGenBuffers(1, &rail_vertex_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, rail_vertex_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof rail_vertices, rail_vertices, GL_STATIC_DRAW);
  glGenBuffers(1, &rail_index_vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rail_index_vbo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof rail_indices, rail_indices, GL_STATIC_DRAW);

  /* set up array buffers for barrier */
  glGenBuffers(1, &barrier_vertex_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, barrier_vertex_vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof barrier_vertices, barrier_vertices, GL_STATIC_DRAW);
  glGenBuffers(1, &barrier_index_vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, barrier_index_vbo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof barrier_indices, barrier_indices, GL_STATIC_DRAW);

  reshape_viewport();
}

static inline void
update_view()
{
  static float rho, theta, margin, ex, ey, controlx, controly, longside, shortside;
  static int mousex, mousey;
  static float dt, nt, t;

  nt = (float)SDL_GetTicks() / 1000.0f;

  dt = nt - t;
  t = nt;

  SDL_GetMouseState(&mousex, &mousey);
  longside = vwidth > vheight? vwidth: vheight;
  shortside = vwidth < vheight? vwidth: vheight;

  margin = (longside - shortside) / 2.0f;
  ex = -((width >= height? clamp((float)mousex, margin, margin + shortside) - margin: (float)mousex) / shortside - 0.5f) * 2.0f;
  ey =  ((width >= height? (float)mousey: clamp((float)mousey, margin, margin + shortside) - margin) / shortside - 0.5f) * 2.0f;

  polar(ex, ey, &rho, &theta);

  euclidean(clamp(rho, -0.45f, 0.45f), theta, &controlx, &controly);

  currentx += (controlx - currentx) * dt / DRIFT_DAMPING;
  currenty += (controly - currenty) * dt / DRIFT_DAMPING;

  ViewProjectionMatrix[12] = currentx * fovaspect;
  ViewProjectionMatrix[13] = currenty * FOV;
}

/* This is necessary to have warpedtime in db_comparator's scope . */
static float warpedtime;

static int db_comparator(const void * av, const void * bv)
{
  const DrawBarrier * a = av, * b = bv;
  float az = fmodf(warpedtime - a->offset, 1.0);
  float bz = fmodf(warpedtime - b->offset, 1.0);
  return az < bz? 1: bz < az? -1: 0;
}

static inline void
draw()
{
  static GLfloat ModelMatrix[4];
  static float time;
  static DrawBarrier barrier;
  static uint32_t i;

  time = (float)SDL_GetTicks() / 1000.0f;
  warpedtime = time / 2.0f; /* TODO: calculate warped time and global speed */

  glClear(GL_COLOR_BUFFER_BIT);

  glUseProgram(rail_program);

  glUniformMatrix4fv(rail_uniform_view_projection_matrix, 1, GL_FALSE, ViewProjectionMatrix);

  rotationmat2(ModelMatrix, 0.35f * time);
  glUniformMatrix2fv(rail_uniform_model_matrix, 1, GL_FALSE, ModelMatrix);

  glBindBuffer(GL_ARRAY_BUFFER, rail_vertex_vbo);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rail_index_vbo);

  glDrawElements(GL_TRIANGLES, sizeof(rail_indices), GL_UNSIGNED_SHORT, NULL);

  qsort(draw_barriers, 4, sizeof(DrawBarrier), db_comparator);

  glBindBuffer(GL_ARRAY_BUFFER, barrier_vertex_vbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, barrier_index_vbo);

  i = 2;
  while (i--) {
    static size_t final_index;
    static float front_position;

    barrier = draw_barriers[i];

    final_index = (size_t)(floor(warpedtime - barrier.offset) + barrier.base_index) % MG3D_TOTAL_BARRIERS;
    front_position = fmodf(warpedtime - barrier.offset, 1.0f) * CLIP;

    rotationmat2(ModelMatrix, barrier.rotspeed * time);

    glUseProgram(barrier_programs[final_index].program);
    glUniformMatrix4fv(barrier_programs[final_index].viewProjectionMatrix, 1, GL_FALSE, ViewProjectionMatrix);
    glUniformMatrix2fv(barrier_programs[final_index].modelMatrix, 1, GL_FALSE, ModelMatrix);

    glUniform1f(barrier_programs[final_index].modelZ, front_position - 0.075f);
    glUniform1f(barrier_programs[final_index].grey,  0.6f);
    glDrawElements(GL_TRIANGLES, sizeof(barrier_indices), GL_UNSIGNED_SHORT, NULL);

    glUniform1f(barrier_programs[final_index].modelZ, front_position);
    glUniform1f(barrier_programs[final_index].grey, 1.0f);
    glDrawElements(GL_TRIANGLES, sizeof(barrier_indices), GL_UNSIGNED_SHORT, NULL);
  }

  SDL_GL_SwapWindow(window);
}

int
main()
{
  uint32_t window_flags;

  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    SDL_Log("error: Failed to initialize SDL with video: %s\n", SDL_GetError());
    return 1;
  }

  window_flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_OPENGL;
#if __ANDROID__ || __IPHONEOS__ || __WINRT__
  window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
#endif
  window = SDL_CreateWindow("mg3d", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 384, 384, window_flags);

  if (window == NULL) {
    SDL_Log("error: Failed to create window: %s", SDL_GetError());
    return 1;
  }

  SDL_ShowCursor(SDL_DISABLE);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
  SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);

  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 0);

  gl_context = SDL_GL_CreateContext(window);
  if (gl_context == NULL) {
    SDL_Log("error: Failed to create OpenGL context: %s", SDL_GetError());
    return 1;
  }

  init();

  while (rendering) {
    handle_SDL_events();

    update_view();

    draw();
  }

  return 0;
}
