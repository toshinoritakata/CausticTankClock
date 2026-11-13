// 3D tank renderer shaders (WGSL). All look-tuning knobs are in the LOOK block of COMMON.
// Units: tank footprint = 1 x 1 (== solver domain), water surface at y = h*relief (~0), floor at y = -D.
// Solver texel (ix,iy) <-> world x = ix/(N-1)-0.5, z = iy/(N-1)-0.5 (glyph top is at -z = far side from the default camera).

export const COMMON = /* wgsl */`
// ===================== LOOK (tune here) =====================
const IOR            : f32 = 1.33;                       // must match solver (nw = 1.33)
const ABSORB         : vec3f = vec3f(0.55, 0.14, 0.07);  // per-channel water absorption per unit length
const SCATTER_COL    : vec3f = vec3f(0.0045, 0.019, 0.023);  // in-scattered water body colour (lit by floor glow)
const FLOOR_ALBEDO   : vec3f = vec3f(0.88, 0.90, 0.92);
const FLOOR_AMBIENT  : f32 = 0.004;
// Diffuse (Lambertian) floor: radiance = albedo x irradiance, and the irradiance is C (relative to the mean, = 1 for a flat surface) x the exposure slider.
// FLOOR_C0 = 0 and FLOOR_POW = 1 make that exactly linear (physical); round 6/7 used C0 0.4 / POW 1.45 (a look curve that blacked out the dimmer floor).
const FLOOR_GAIN     : f32 = 0.15;                       // irradiance -> scene radiance scale (an exposure constant, not a shape)
const FLOOR_C0       : f32 = 0.0;                        // look-only offset subtracted from C (0 = physical)
const FLOOR_POW      : f32 = 1.0;                        // look-only power on C (1 = physical)
const FLOOR_LMAX     : f32 = 3.0;                        // soft cap: l -> FLOOR_LMAX*tanh(l/FLOOR_LMAX) (<2% change below l~0.9 = C~2.9): transition / disturb cusps (C 40-60) cannot flash the bloom
const EDGE_LIT_REF   : f32 = 2.5;                        // glass edges sample floorShade x this before the l/(1+l) compression (lower = less saturated compression: dark floor -> dark edge, lit floor -> bright)
const LIGHT_COL      : vec3f = vec3f(1.0, 0.95, 0.86);
const TOP_GAIN       : f32 = 2.0;                        // overhead soft bars (reflected in water): peak radiance; x F0 0.02 -> they sit well under the caustic digits
const TOP_W          : f32 = 0.07;                       // bar width: gaussian sigma*sqrt2 across the bar (tangent-plane units at the key direction, ~4 deg) -> smooth falloff, no hard edge
const TOP_BAR_A      : vec4f = vec4f(-0.34, 0.08, 0.58, 0.0);  // bar A (key frame): centre q, half length, axis (0 = along q.y)
const TOP_BAR_B      : vec4f = vec4f(0.18, 0.78, 0.30, 1.0);   // bar B: placed so the reflection over the digits misses both bars for presets 1/2 at elev 35..90 (az 30)
const REFL_OVER_FLOOR: f32 = 3.0;                        // water env reflection x 1/(1 + this x luminance of the floor seen through it): the overhead bars never wash a grey band over the lit digits
const REFL_ROUGH     : f32 = 0.035;                      // water reflection blur: 4 extra env taps this far around the mirror direction (fine ripples)
const STRIP_GAIN     : f32 = 4.0;                        // side / back strip lights
const STRIP_WARM     : vec3f = vec3f(0.877, 0.778, 0.675);  // warm side strip colour (old 1,0.78,0.55 pulled ~55% toward neutral: the glass reflection no longer reads cream/orange)
const F0_WATER       : f32 = 0.02;
const SUN_POW        : f32 = 900.0;
const SUN_INT        : f32 = 18.0;
const GLASS_REFL     : f32 = 0.25;                       // glass wall reflection colour strength (linear)
const GLASS_ALPHA    : f32 = 1.1;                        // glass wall opacity from the Fresnel term (how much it hides what is behind)
const GLASS_TOP      : f32 = 2.0;                        // overhead lights as seen in the glass (x TOP_GAIN = 4 as accepted; the old soft box was 6 over a much larger area)
const GLASS_T        : f32 = 0.014;                      // pane thickness (world units; tank = 1 wide): rim / corner edge faces are real geometry (vsRim)
const BASE_T         : f32 = 0.02;                       // bottom plate thickness: the tank stands on the table (y = -D - BASE_T)
const EDGE_COL       : vec3f = vec3f(0.46, 0.78, 0.68);  // glass edge colour (long path through the pane: pale green / teal)
const EDGE_PIPE      : f32 = 1.3;                        // light from the lit water / floor carried inside the pane to its edge faces
const EDGE_FALL      : f32 = 0.9;                        // piped light falloff with height above the floor (per unit)
const EDGE_AMB       : f32 = 0.006;                      // light the whole pane collects from the glowing water body (x (0.3 + avgC), more near the floor): edges never read as black bars
const EDGE_KEY       : f32 = 0.12;                       // key light falling on the pane faces, trapped and carried to the edges (x |pane normal . light|): sun-side panes glow brighter
const EDGE_SPEC      : f32 = 0.5;                        // soft key-light glint on the polished edge faces (moves along the rim with the view)
const RIM_MINPX      : f32 = 0.8;                        // edge faces never thinner than 2 x this (framebuffer px): the top rim stays a stable thin line at grazing pitch
const EDGE_MIN       : f32 = 0.02;                       // minimum glow of the edge faces (light trapped in the glass always exits at the edge)
const UNDER_GAIN     : f32 = 0.8;                        // water underside (TIR mirror + Snell's window) overall level
const WALL_TOP       : f32 = 0.15;                        // glass wall height above water level
const TABLE_GLOSS    : f32 = 0.2;
const TABLE_TOP      : f32 = 0.3;                        // overhead lights as seen in the table (x TOP_GAIN = 0.6 as accepted; sharp strips there read as stray grey bands)
const TABLE_FADE0    : f32 = 2.2;                        // table -> background blend: starts at this distance from the tank centre ...
const TABLE_FADE1    : f32 = 6.5;                        // ... complete here (table edge is at 7)
const SPILL_GAIN     : f32 = 0.09;                        // light spilled on table around tank
const PHASE_KNEE     : f32 = 1.0;                        // god-ray phase factor: unchanged below, soft-limited above (HG forward peak at g=0.5 is ~4.5x -> blows out when looking into the light)
const PHASE_MAX      : f32 = 1.8;
const RAY_FLOOR      : f32 = 0.6;                        // god rays: volume level subtracted before scattering (removes the uniform fog, keeps the shaft contrast)
const RAY_REGAIN     : f32 = 1.3;                        // gain applied after the floor subtraction (restores shaft peak strength)
const VIEW_SLOPE     : f32 = 0.3;                       // LOOK ONLY: scales surface slopes used for refraction shading (1 = physical; low keeps the digits legible)
const REFL_SLOPE     : f32 = 0.6;                       // LOOK ONLY: slope scale for the reflection normal (near physical: highlights follow the waves)
const RAY_SURF       : f32 = 0.12;                      // god rays seen down through the surface (shafts end-on integrate into a grey fog over the floor); side view unchanged
// ============================================================

struct U { vp:mat4x4f, eye:vec4f, fwd:vec4f, rgt:vec4f, upv:vec4f, a:vec4f, b:vec4f, c:vec4f, g:vec4f, g2:vec4f, sun:vec4f, lw:vec4f, m:vec4f }
// sun.xyz = unit vector TOWARD the light (world x, up, z); lw.xyz = unit light travel direction INSIDE the water (refracted); m = (meshM, flat, 0, 0)
// god rays: g = (on, gain, sigma, aniso g)   g2 = (steps, noise amp, NZ, 0)
// c.x = floor glow (anim: rises as letter sharpens)
// a = (D, relief, expo, N)   b = (avgC, glassOn, resX, resY)   rgt.w = aspect, fwd.w = tan(fovy/2)
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var<storage,read> hT:array<f32>;
@group(0) @binding(2) var<storage,read> hxb:array<f32>;
@group(0) @binding(3) var<storage,read> hyb:array<f32>;
@group(0) @binding(4) var<storage,read> Cb:array<f32>;
@group(0) @binding(5) var vol:texture_3d<f32>;
@group(0) @binding(6) var vsm:sampler;

fn bilH(uv:vec2f)->f32 {
  let n=i32(u.a.w); let q=clamp(uv,vec2f(0.0),vec2f(1.0))*f32(n-1); let f=floor(q); let a=q-f;
  let x0=min(i32(f.x),n-1); let y0=min(i32(f.y),n-1); let x1=min(x0+1,n-1); let y1=min(y0+1,n-1);
  return mix(mix(hT[y0*n+x0],hT[y0*n+x1],a.x),mix(hT[y1*n+x0],hT[y1*n+x1],a.x),a.y);
}
fn bil(which:i32, uv:vec2f)->f32 {
  let n=i32(u.a.w); var q=clamp(uv,vec2f(0.0),vec2f(1.0))*f32(n-1);
  if (which>=2 && u.m.z>0.5) { q=floor(q+vec2f(0.5)); }   // 'nearest' caustic sampling: chunky texels
  let f=floor(q); let a=q-f;
  let x0=min(i32(f.x),n-1); let y0=min(i32(f.y),n-1); let x1=min(x0+1,n-1); let y1=min(y0+1,n-1);
  var v00=0.0; var v10=0.0; var v01=0.0; var v11=0.0;
  if (which==0) { v00=hxb[y0*n+x0]; v10=hxb[y0*n+x1]; v01=hxb[y1*n+x0]; v11=hxb[y1*n+x1]; }
  else if (which==1) { v00=hyb[y0*n+x0]; v10=hyb[y0*n+x1]; v01=hyb[y1*n+x0]; v11=hyb[y1*n+x1]; }
  else { let o=(which-2)*n*n; v00=Cb[o+y0*n+x0]; v10=Cb[o+y0*n+x1]; v01=Cb[o+y1*n+x0]; v11=Cb[o+y1*n+x1]; }   // which 2/3/4 = caustic plane G/R/B
  return mix(mix(v00,v10,a.x), mix(v01,v11,a.x), a.y);
}

// local water level at a border point of the tank (uv on the tank edge), exactly as the water mesh draws it: linear between mesh-lattice vertices (m x m), vertices at the edge
fn wallH(s:f32, fixedC:f32, alongX:bool)->f32 {
  let m=max(2.0,u.m.x); let t=clamp(s,0.0,1.0)*(m-1.0); let i0=floor(t); let a=t-i0; let i1=min(i0+1.0,m-1.0);
  var p0=vec2f(fixedC,i0/(m-1.0)); var p1=vec2f(fixedC,i1/(m-1.0));
  if (alongX) { p0=vec2f(i0/(m-1.0),fixedC); p1=vec2f(i1/(m-1.0),fixedC); }
  return mix(bilH(p0),bilH(p1),a)*u.a.y;
}
// rotate so the sun direction maps to straight up (identity at elevation 90): the overhead key soft box follows the light
fn keyRot(d:vec3f)->vec3f {
  let s=u.sun.xyz; let hl=length(s.xz);
  if (hl<1e-4) { return d; }
  let eh=vec3f(s.x,0.0,s.z)/hl; let ep=vec3f(-eh.z,0.0,eh.x);
  let a=dot(d,eh); let b=d.y; let p=dot(d,ep);
  let a2=a*s.y-b*hl; let b2=a*hl+b*s.y;
  return a2*eh+vec3f(0.0,b2,0.0)+p*ep;
}

// ---------- god rays: ray-march of the caustic light volume along a view ray inside the water ----------
fn ign(p:vec2f)->f32 { return fract(52.9829189*fract(dot(p,vec2f(0.06711056,0.00583715)))); }
fn h13(p:vec3f)->f32 { var q=fract(p*0.1031); q+=dot(q,q.zyx+31.32); return fract((q.x+q.y)*q.z); }
fn vnoise(p:vec3f)->f32 {
  let i=floor(p); let f=fract(p); let w=f*f*(3.0-2.0*f);
  return mix(mix(mix(h13(i),h13(i+vec3f(1,0,0)),w.x), mix(h13(i+vec3f(0,1,0)),h13(i+vec3f(1,1,0)),w.x), w.y),
             mix(mix(h13(i+vec3f(0,0,1)),h13(i+vec3f(1,0,1)),w.x), mix(h13(i+vec3f(0,1,1)),h13(i+vec3f(1,1,1)),w.x), w.y), w.z);
}
fn volSample(p:vec3f)->f32 {
  let D=u.a.x; let nz=u.g2.z; let z=clamp(-p.y,0.0,D)/D; let w=(z*(nz-1.0)+0.5)/nz;
  return textureSampleLevel(vol,vsm,vec3f(p.x+0.5,p.z+0.5,w),0.0).r;
}
// P: start (on the water surface / glass), r: unit direction of travel (towards the far side), tEnd: length of the in-water segment
fn godRays(P:vec3f, r:vec3f, tEnd:f32, px:vec2f)->vec3f {
  if (u.g.x<0.5 || tEnd<=1e-4) { return vec3f(0.0); }
  let S=max(1,i32(u.g2.x)); let ds=min(tEnd,3.0)/f32(S); let sig=u.g.z; let ga=u.g.w;   // in-tank path <= ~1.8; callers' t / tw blow up for near-horizontal rays (1e9 sentinel) -> one-pixel HDR fireflies
  let Lw=u.lw.xyz; let cs=-dot(Lw,r);   // light travel dir inside the water (refracted); cos(scatter angle) = dot(L, -r)
  let hg=(1.0-ga*ga)/pow(max(1.0+ga*ga-2.0*ga*cs,1e-3),1.5);
  var phase=0.3+0.7*hg;
  if (phase>PHASE_KNEE) { phase=PHASE_KNEE+(PHASE_MAX-PHASE_KNEE)*tanh((phase-PHASE_KNEE)/(PHASE_MAX-PHASE_KNEE)); }
  let j=ign(px);
  // transmittance (view path + light path) advanced multiplicatively: both are exp of a linear function of s
  let ilz=1.0/max(-Lw.y,0.2);   // light path length per unit depth
  let kv=ABSORB*0.5+vec3f(sig)+ABSORB*0.6*ilz*max(-r.y,0.0);   // max: an upward-bent ray (steep wave at grazing view) must not turn the decay into growth (HDR fireflies)
  let stepT=exp(-kv*ds);
  var T=exp(-ABSORB*0.6*ilz*clamp(-P.y,0.0,8.0)-(ABSORB*0.5+vec3f(sig))*0.0-kv*(j*ds));
  var acc=vec3f(0.0);
  for (var i=0; i<S; i++) {
    let p=P+r*((f32(i)+j)*ds);
    acc+=max(min(volSample(p),14.0)-RAY_FLOOR,0.0)*RAY_REGAIN*T*mix(0.25,1.0,smoothstep(0.0,0.3,-p.y));   // shallow light is ~uniform (no shafts yet): de-emphasise the glowing band under the surface
    T*=stepT;
  }
  return acc*ds*sig*u.g.y*phase*LIGHT_COL*vec3f(0.97,1.0,1.0);
}


// soft bar light in the key tangent plane: gaussian across, smooth taper + brightness gradient along (bar = centre.xy, half length, axis)
fn softBar(q:vec2f, bar:vec4f)->f32 {
  var d = q - bar.xy; if (bar.w > 0.5) { d = d.yx; }
  let t = abs(d.y)/bar.z;
  return exp(-d.x*d.x/(TOP_W*TOP_W)) * (1.0 - smoothstep(0.55, 1.0, t)) * (1.0 - 0.45*t*t);
}
// procedural dark-room environment: soft boxes / strips on a near-black gradient
fn env(d:vec3f, top:f32)->vec3f {   // top: gain of the overhead lights (1 = water)
  var c = mix(vec3f(0.003,0.004,0.006), vec3f(0.018,0.026,0.045), clamp(d.y*0.5+0.5,0.0,1.0));
  let dk = keyRot(d);
  if (dk.y > 0.12) {
    let q = dk.xz / dk.y;
    let b = softBar(q, TOP_BAR_A) + softBar(q, TOP_BAR_B);   // two soft studio bars off the key direction: never a uniform sheet
    c += TOP_GAIN * top * b * vec3f(1.0, 0.96, 0.9);
    let r = smoothstep(0.10, 0.0, abs(length(q - vec2f(0.1,-0.15)) - 1.5)) * 0.25;
    c += r * vec3f(0.5,0.7,1.0);
  }
  let ax = abs(d.x);
  if (ax > 0.15) {
    let q = vec2f(d.y, d.z) / ax;
    let s = smoothstep(0.12, 0.0, max(abs(q.x - 0.35) - 0.9, abs(q.y - 0.25) - 0.05));
    c += STRIP_GAIN * s * select(STRIP_WARM, vec3f(0.55,0.75,1.0), d.x < 0.0);
  }
  if (d.z < -0.15) {
    let q = vec2f(d.x, d.y) / (-d.z);
    let s = smoothstep(0.12, 0.0, max(abs(q.x) - 1.6, abs(q.y - 0.55) - 0.035));
    c += 2.5 * s * vec3f(0.9,0.95,1.0);
  }
  return c;
}

fn floorShade(p:vec2f)->vec3f {
  let c = vec3f(bil(3, p + 0.5), bil(2, p + 0.5), bil(4, p + 0.5));   // caustic per colour plane (R, G, B: dispersion fringes on the filaments)
  let k = max(c-vec3f(FLOOR_C0),vec3f(0.0)) * u.a.z;
  let l = FLOOR_LMAX * tanh(min(pow(k, vec3f(FLOOR_POW)) * FLOOR_GAIN / FLOOR_LMAX, vec3f(10.0))) / FLOOR_GAIN;   // (tanh arg clamped: no exp overflow)
  let edge = smoothstep(0.0, 0.16, 0.5 - max(abs(p.x), abs(p.y)));
  let vig = mix(0.5, 1.0, edge);
  let glow = LIGHT_COL * u.c.x * 0.03 * exp(-dot(p,p)*5.0);
  return FLOOR_ALBEDO * (FLOOR_AMBIENT + LIGHT_COL * l * FLOOR_GAIN + glow) * vig;
}
// background radiance for view direction rd at screen ndc (the bg pass, and the far table fading into it)
fn bgCol(rd:vec3f, ndc:vec2f)->vec3f {
  var c = vec3f(0.0025,0.003,0.0045)*(1.0+4.0*pow(1.0-abs(rd.y),4.0));
  c += vec3f(0.004,0.007,0.009)*u.b.x*exp(-length(ndc)*1.6);
  c *= 1.0-0.45*dot(ndc*0.6,ndc*0.6);
  let sdot=dot(rd,normalize(u.sun.xyz));  // small sun disc (visible from side views)
  c += LIGHT_COL*6.0*smoothstep(0.9990,0.9996,sdot)+LIGHT_COL*0.05*pow(max(sdot,0.0),64.0);
  return c;
}
fn waterTintAt(depthFrac:f32)->vec3f { return SCATTER_COL * (0.35 + 0.7*u.b.x) * exp(-depthFrac*1.6); }
`;

export const SCENE_WGSL = COMMON + /* wgsl */`
struct FS { @builtin(position) pos:vec4f, @location(0) w:vec3f, @location(1) uv:vec2f, @location(2) lc:vec3f }

// ---------- water surface (displaced grid) ----------
@vertex fn vsWater(@builtin(vertex_index) vi:u32) -> FS {
  let m=u32(u.m.x); let cells=m-1u; let cell=vi/6u; let c=vi%6u;
  var cx=array<u32,6>(0u,1u,0u,1u,1u,0u); var cy=array<u32,6>(0u,0u,1u,0u,1u,1u);
  let ix=cell%cells+cx[c]; let iy=cell/cells+cy[c];
  let uv=vec2f(f32(ix),f32(iy))/f32(m-1u);   // vertices on an m x m lattice; border vertices sit exactly at uv 0/1 (tank edge); shared by neighbours -> no cracks
  let h=bilH(uv)*u.a.y;
  let w=vec3f(uv.x-0.5, h, uv.y-0.5);
  var o:FS; o.pos=u.vp*vec4f(w,1.0); o.w=w; o.uv=uv; o.lc=vec3f(0.0); return o;
}
@fragment fn fsWater(i:FS) -> @location(0) vec4f {
  let R=u.a.y; let D=u.a.x;
  let fnr=cross(dpdx(i.w),dpdy(i.w));
  var gx=bil(0,i.uv)*R; var gz=bil(1,i.uv)*R;   // physical slopes
  if (u.m.y>0.5) { let fo=select(-fnr,fnr,fnr.y>0.0); let fy=max(abs(fo.y),1e-9); gx=-fo.x/fy; gz=-fo.z/fy; }
  let n=normalize(vec3f(-gx*VIEW_SLOPE, 1.0, -gz*VIEW_SLOPE));    // refraction normal
  let nr=normalize(vec3f(-gx*REFL_SLOPE, 1.0, -gz*REFL_SLOPE));   // reflection normal
  let P=vec3f(clamp(i.w.xz,vec2f(-0.5),vec2f(0.5)),i.w.y).xzy; let rd=normalize(P-u.eye.xyz);   // clamp: MSAA evaluates at the pixel centre, which can extrapolate past the tank edge -> negative wall distances -> exp(+ABSORB*..) fireflies
  let cosi=-dot(rd,n);
  if (cosi < 0.0) { // seen from below (through the side glass): TIR mirror of the lit floor / inner walls + Snell's window to the room above
    let nb=-nr; let ci=clamp(dot(rd,nr),0.0,1.0);   // cos of incidence (water side)
    let st2=IOR*IOR*(1.0-ci*ci);                      // sin^2 of the transmitted angle
    var Fu=1.0; var trans=vec3f(0.0);
    if (st2<1.0) { let ct=sqrt(1.0-st2); Fu=F0_WATER+(1.0-F0_WATER)*pow(1.0-ct,5.0); trans=env(refract(rd,nb,IOR),1.0)*(1.0-Fu); }
    let rf=reflect(rd,nb);   // goes back down into the water
    var mir=vec3f(0.0);
    let tf=min(-D-P.y,0.0)/min(rf.y,-1e-3); let Fh=P+rf*tf;
    if (rf.y<-1e-3 && abs(Fh.x)<=0.5 && abs(Fh.z)<=0.5) {
      mir=floorShade(Fh.xz)*exp(-ABSORB*tf)+waterTintAt(0.0)*(1.0-exp(-tf*0.9));
    } else {
      let tx=select(1e9,(sign(rf.x)*0.5-P.x)/rf.x,abs(rf.x)>1e-5); let tz=select(1e9,(sign(rf.z)*0.5-P.z)/rf.z,abs(rf.z)>1e-5);
      let tw=max(min(tx,tz),0.0); let hw=P+rf*tw;
      mir=waterTintAt(clamp(-hw.y/D,0.0,1.0))*1.2*exp(-ABSORB*tw*0.5)+vec3f(0.004,0.010,0.012);
    }
    // path from the surface point back to the side glass the viewer looks through (still in water)
    let bx=select(1e9,(sign(-rd.x)*0.5-P.x)/(-rd.x),abs(rd.x)>1e-5); let bz=select(1e9,(sign(-rd.z)*0.5-P.z)/(-rd.z),abs(rd.z)>1e-5);
    let tb=clamp(min(bx,bz),0.0,2.0);
    let col=(mir*Fu+trans)*exp(-ABSORB*tb)*UNDER_GAIN;
    return vec4f(col,1.0);
  }
  let F=F0_WATER+(1.0-F0_WATER)*pow(1.0-clamp(-dot(rd,nr),0.0,1.0),5.0);
  let r=refract(rd,n,1.0/IOR);
  var col=vec3f(0.0);
  let t=min(-D-P.y,0.0)/min(r.y,-1e-3);   // min: a disturb trough can dip below the floor plane -> t<0 -> negative col (and refl / (1 + k*lum) -> 1/0 fireflies)
  let Fh=P+r*t;
  if (abs(Fh.x)<=0.5 && abs(Fh.z)<=0.5) {
    col=floorShade(Fh.xz)*exp(-ABSORB*t);
    col+=waterTintAt(0.0)*(1.0-exp(-t*0.9));
    col+=godRays(P,r,t,i.pos.xy)*RAY_SURF;
  } else { // ray leaves through a side wall: shade the inner wall
    let tx=select(1e9,(sign(r.x)*0.5-P.x)/r.x,abs(r.x)>1e-5);
    let tz=select(1e9,(sign(r.z)*0.5-P.z)/r.z,abs(r.z)>1e-5);
    let tw=max(min(tx,tz),0.0); let hw=P+r*tw;
    let hs=bilH(clamp(hw.xz+vec2f(0.5),vec2f(0.0),vec2f(1.0)))*R;
    let df=clamp((hs-hw.y)/D,0.0,1.0);   // depth below the LOCAL surface at the wall point
    col=waterTintAt(df)*1.2*exp(-ABSORB*tw*0.5)+vec3f(0.004,0.010,0.012);
    col+=godRays(P,r,tw,i.pos.xy)*RAY_SURF;
  }
  let rr=reflect(rd,nr);
  let t1=normalize(cross(rr,select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(rr.y)>0.9))); let t2=cross(rr,t1);
  var refl=env(rr,1.0)*0.4+(env(normalize(rr+t1*REFL_ROUGH),1.0)+env(normalize(rr-t1*REFL_ROUGH),1.0)+env(normalize(rr+t2*REFL_ROUGH),1.0)+env(normalize(rr-t2*REFL_ROUGH),1.0))*0.15;
  refl*=1.0/(1.0+REFL_OVER_FLOOR*max(dot(col,vec3f(0.2126,0.7152,0.0722)),0.0));
  let sd=normalize(u.sun.xyz);
  refl+=SUN_INT*pow(max(dot(rr,sd),0.0),SUN_POW)*LIGHT_COL;
  let outv=col*(1.0-F)+refl*F;
  return vec4f(outv,1.0);
}

// ---------- tank floor (visible through side glass) ----------
@vertex fn vsFloor(@builtin(vertex_index) vi:u32) -> FS {
  var cx=array<f32,6>(0.0,1.0,0.0,1.0,1.0,0.0); var cy=array<f32,6>(0.0,0.0,1.0,0.0,1.0,1.0);
  let w=vec3f(cx[vi]-0.5,-u.a.x,cy[vi]-0.5);
  var o:FS; o.pos=u.vp*vec4f(w,1.0); o.w=w; o.uv=vec2f(cx[vi],cy[vi]); o.lc=vec3f(0.0); return o;
}
@fragment fn fsFloor(i:FS) -> @location(0) vec4f {
  let rd=normalize(i.w-u.eye.xyz);
  let t=u.a.x*1.3+0.15;
  var col=floorShade(i.w.xz)*exp(-ABSORB*t)+waterTintAt(0.0)*(1.0-exp(-t*0.9));
  return vec4f(col,1.0);
}

// ---------- table ----------
@vertex fn vsTable(@builtin(vertex_index) vi:u32) -> FS {
  var cx=array<f32,6>(0.0,1.0,0.0,1.0,1.0,0.0); var cy=array<f32,6>(0.0,0.0,1.0,0.0,1.0,1.0);
  let w=vec3f((cx[vi]-0.5)*14.0,-u.a.x-0.02,(cy[vi]-0.5)*14.0);
  var o:FS; o.pos=u.vp*vec4f(w,1.0); o.w=w; o.uv=vec2f(0.0); o.lc=vec3f(0.0); return o;
}
@fragment fn fsTable(i:FS) -> @location(0) vec4f {
  let p=i.w.xz; let rd=normalize(i.w-u.eye.xyz);
  let q=max(abs(p)-0.52,vec2f(0.0)); let d=length(q);
  let cosv=clamp(-rd.y,0.0,1.0);
  let F=0.04+0.96*pow(1.0-cosv,5.0);
  // slightly blurred reflection: sample env around the mirror direction
  let rr=reflect(rd,vec3f(0.0,1.0,0.0));
  var refl=env(rr,TABLE_TOP)*0.5+env(normalize(rr+vec3f(0.06,0.0,0.0)),TABLE_TOP)*0.25+env(normalize(rr-vec3f(0.06,0.0,0.0)),TABLE_TOP)*0.25;
  var col=vec3f(0.006,0.0065,0.008)+refl*F*TABLE_GLOSS*0.35;
  let spill=exp(-d*5.0)*SPILL_GAIN*(0.2+u.b.x);
  col+=spill*vec3f(0.30,0.55,0.62);
  col+=exp(-d*28.0)*0.08*vec3f(0.6,0.85,0.9)*(0.2+u.b.x);
  col*=exp(-length(p)*0.22);
  // dark contact shadow right at the glass base
  col*=mix(0.35,1.0,smoothstep(0.0,0.02,d));
  // far table dissolves into the background (no horizon line where the 14 x 14 table ends)
  let ndc=vec2f(i.pos.x/u.b.z*2.0-1.0, 1.0-i.pos.y/u.b.w*2.0);
  col=mix(col, bgCol(rd, ndc), smoothstep(TABLE_FADE0, TABLE_FADE1, length(p)));
  return vec4f(col,1.0);
}

// ---------- glass walls ----------
@vertex fn vsGlass(@builtin(vertex_index) vi:u32, @builtin(instance_index) ii:u32) -> FS {
  var cx=array<f32,6>(0.0,1.0,0.0,1.0,1.0,0.0); var cy=array<f32,6>(0.0,0.0,1.0,0.0,1.0,1.0);
  let s=cx[vi]-0.5; let y=mix(-u.a.x-BASE_T, WALL_TOP, cy[vi]);
  var w=vec3f(0.0); var lcx=s;
  if (ii==0u) { w=vec3f(s,y,0.5); } else if (ii==1u) { w=vec3f(s,y,-0.5); }
  else if (ii==2u) { w=vec3f(0.5,y,s); } else { w=vec3f(-0.5,y,s); }
  var o:FS; o.pos=u.vp*vec4f(w,1.0); o.w=w; o.uv=vec2f(0.0); o.lc=vec3f(lcx,y,f32(ii)); return o;
}
@fragment fn fsGlass(i:FS) -> @location(0) vec4f {
  if (u.b.y < 0.5) { discard; }
  let ii=u32(i.lc.z + 0.5);
  var n=vec3f(0.0,0.0,1.0);
  if (ii==1u) { n=vec3f(0.0,0.0,-1.0); } else if (ii==2u) { n=vec3f(1.0,0.0,0.0); } else if (ii==3u) { n=vec3f(-1.0,0.0,0.0); }
  let rd=normalize(i.w-u.eye.xyz);
  if (dot(n,rd)>0.0) { n=-n; }   // far pane (seen from inside)
  let cosv=clamp(-dot(n,rd),0.0,1.0);
  let F=0.05+0.95*pow(1.0-cosv,4.0);
  let refl=env(reflect(rd,n),GLASS_TOP)*F*GLASS_REFL;
  // local waterline: the water mesh height at this wall point (wall uv: ii 0/1 -> z=+/-0.5 (uv.y=1/0), ii 2/3 -> x=+/-0.5 (uv.x=1/0); the other uv = lc.x+0.5)
  let alongX=(ii<2u); let fixedC=select(0.0,1.0,(ii==0u)||(ii==2u));
  let hw=wallH(i.lc.x+0.5,fixedC,alongX);
  let under=(1.0-smoothstep(hw-0.003,hw+0.003,i.lc.y))*step(-u.a.x,i.lc.y);   // water behind the pane (not the base plate strip)
  let wa=under*0.30;
  let inside=waterTintAt(clamp((hw-i.lc.y)/u.a.x,0.0,1.0))*(2.2)*wa;
  let men=exp(-pow((i.lc.y-hw)/0.004,2.0));   // faint meniscus line on the waterline
  // faint vertical sheen band
  let sheen=smoothstep(0.35,0.0,abs(i.lc.x-0.18))*0.02*F;
  var rgb=refl+inside+sheen+vec3f(0.35,0.55,0.62)*men*0.35;
  let a=clamp(F*GLASS_ALPHA+wa+men*0.25,0.0,1.0);
  if (u.g.x>0.5 && under>0.5) { // side view: march the water column behind the entry glass
    var no=vec3f(0.0,0.0,1.0);
    if (ii==1u) { no=vec3f(0.0,0.0,-1.0); } else if (ii==2u) { no=vec3f(1.0,0.0,0.0); } else if (ii==3u) { no=vec3f(-1.0,0.0,0.0); }
    if (dot(no,rd)<0.0) {
      let D=u.a.x; let q=i.w;
      let tx=select(1e9,select((-0.5-q.x)/rd.x,(0.5-q.x)/rd.x,rd.x>0.0),abs(rd.x)>1e-5);
      let tz=select(1e9,select((-0.5-q.z)/rd.z,(0.5-q.z)/rd.z,rd.z>0.0),abs(rd.z)>1e-5);
      let ty=select(1e9,(-D-q.y)/rd.y,rd.y<-1e-5);   // floor
      var te=min(tx,min(tz,ty));
      // the ray leaves the water through the real heightfield: march until it passes above the surface, then bisect
      var ta=0.0; var tb=-1.0; let NS=16;
      for (var k=1; k<=NS; k++) { let tt=te*f32(k)/f32(NS); let pp=q+rd*tt; if (pp.y>bilH(clamp(pp.xz+vec2f(0.5),vec2f(0.0),vec2f(1.0)))*u.a.y) { tb=tt; break; } ta=tt; }
      if (tb>0.0) { for (var k=0; k<6; k++) { let tm=0.5*(ta+tb); let pp=q+rd*tm; if (pp.y>bilH(clamp(pp.xz+vec2f(0.5),vec2f(0.0),vec2f(1.0)))*u.a.y) { tb=tm; } else { ta=tm; } } te=0.5*(ta+tb); }
      let ins=godRays(q,rd,te,i.pos.xy);
      return vec4f(rgb+ins, a);
    }
  }
  return vec4f(rgb, a);
}

// ---------- glass thickness: edge faces of the panes and the base plate (opaque, drawn before the transparent panes) ----------
// instances 0..3: wall boxes (pane mid-plane at +/-0.5, thickness GLASS_T; the z walls run past the corners, the x walls butt inside them):
//   top rim + one corner post (12 verts used). instance 4: the 4 side faces of the base plate (24 verts). lc = face normal, uv = pane normal (xz; 0 = the horizontal base plate).
@vertex fn vsRim(@builtin(vertex_index) vi:u32, @builtin(instance_index) ii:u32) -> FS {
  var cx=array<f32,6>(0.0,1.0,0.0,1.0,1.0,0.0); var cy=array<f32,6>(0.0,0.0,1.0,0.0,1.0,1.0);
  let face=vi/6u; let a=cx[vi%6u]; let b=cy[vi%6u];
  let D=u.a.x; let ht=0.5*GLASS_T; let y0=-D-BASE_T;
  var w=vec3f(0.0); var nl=vec3f(0.0,1.0,0.0); var pn=vec2f(0.0);
  if (ii<4u) {
    if (face>1u) { var o:FS; o.pos=vec4f(0.0,0.0,2.0,1.0); o.w=vec3f(0.0); o.uv=vec2f(0.0); o.lc=nl; return o; }   // degenerate (clipped)
    // face 0: the pane's top rim = centre line A..B along the pane, widened across it by +/-ht.
    // face 1: the corner post of corner ii (one per corner, both panes' ends together: a single joint seen from any yaw) = the vertical corner line,
    //         widened to the projected pane thickness (the larger of the two panes' GLASS_T offsets: never wider than the pane it joins). No end caps on the post:
    //         it ends flush with the rim top and the base (rims cap only sub-pixel at their ends, to close the corner seam).
    // Drawn as screen-space ribbons: true projected width, but never thinner than RIM_MINPX (a grazing face would go sub-pixel and vanish / shimmer)
    let L=select(0.5-ht,0.5+ht,ii<2u); let sd=select(-0.5,0.5,ii==0u||ii==2u);
    var A=vec3f(-L,WALL_TOP,sd); var B=vec3f(L,WALL_TOP,sd); var d1=vec3f(0.0,0.0,ht); var d2=d1;
    if (ii<2u) { nl=vec3f(0.0,1.0,0.0); pn=vec2f(0.0,1.0); } else { A=vec3f(sd,WALL_TOP,-L); B=vec3f(sd,WALL_TOP,L); d1=vec3f(ht,0.0,0.0); d2=d1; nl=vec3f(0.0,1.0,0.0); pn=vec2f(1.0,0.0); }
    if (face==1u) {
      let cxz=vec2f(select(-0.5,0.5,(ii&1u)==0u), select(-0.5,0.5,ii<2u));
      A=vec3f(cxz.x,y0,cxz.y); B=vec3f(cxz.x,WALL_TOP,cxz.y); d1=vec3f(ht,0.0,0.0); d2=vec3f(0.0,0.0,ht);
      nl=vec3f(cxz.x,-1.0,cxz.y); pn=normalize(cxz);   // lc.y = -1 flags a corner post; lc.xz = the corner (fsRim picks the end face toward the viewer)
    }
    let side=b*2.0-1.0; let P=mix(A,B,a);
    let hr=0.5*u.b.zw;   // NDC -> pixels
    let cA=u.vp*vec4f(A,1.0); let cB=u.vp*vec4f(B,1.0); let cP=u.vp*vec4f(P,1.0); let c1=u.vp*vec4f(P+d1,1.0); let c2=u.vp*vec4f(P+d2,1.0);
    let sA=cA.xy/max(cA.w,1e-4)*hr; let sB=cB.xy/max(cB.w,1e-4)*hr; let sP=cP.xy/max(cP.w,1e-4)*hr;
    let s1=c1.xy/max(c1.w,1e-4)*hr; let s2=c2.xy/max(c2.w,1e-4)*hr;
    var ld=sB-sA; if (dot(ld,ld)<1e-8) { ld=vec2f(1.0,0.0); } ld=normalize(ld); let pp=vec2f(-ld.y,ld.x);
    let wpx=max(max(abs(dot(s1-sP,pp)),abs(dot(s2-sP,pp))),RIM_MINPX);
    let cap=select(min(wpx,RIM_MINPX),0.0,face==1u);   // rims: sub-pixel overlap at the corners (no seam); posts: none (flush top / bottom)
    let off=pp*wpx*side+ld*cap*(a*2.0-1.0);
    var o:FS; o.pos=vec4f(cP.xy+off/hr*cP.w,cP.z,cP.w); o.w=P+select(d1,(d1+d2)*0.5,face==1u)*side; o.uv=pn; o.lc=nl; return o;
  } else {
    let H=0.5-ht; let e=select(-1.0,1.0,(face&1u)==0u); let y=mix(y0,-D,b); let s=mix(-H,H,a);
    if (face<2u) { w=vec3f(e*H,y,s); nl=vec3f(e,0.0,0.0); } else { w=vec3f(s,y,e*H); nl=vec3f(0.0,0.0,e); }
  }
  var o:FS; o.pos=u.vp*vec4f(w,1.0); o.w=w; o.uv=pn; o.lc=nl; return o;
}
fn edgeLum(c:vec3f)->f32 { let l=dot(c,vec3f(0.3,0.55,0.15))*EDGE_LIT_REF; return l/(1.0+l); }   // compressed: bright digits never blow the edge out
@fragment fn fsRim(i:FS) -> @location(0) vec4f {
  if (u.b.y < 0.5) { discard; }
  let D=u.a.x; let p=i.w; let rd=normalize(p-u.eye.xyz);
  var n=normalize(i.lc); if (dot(n,rd)>0.0) { n=-n; }
  if (i.lc.y < -0.5) {   // corner post: the two panes' end faces meet here; shade the one turned more toward the viewer (real normal: grazing / face-on variation as in round 5)
    let nx=vec3f(sign(i.lc.x),0.0,0.0); let nz=vec3f(0.0,0.0,sign(i.lc.z));
    n=select(nz,nx,-dot(nx,rd)>-dot(nz,rd));
  }
  let cosv=clamp(-dot(n,rd),0.0,1.0);
  let F=0.04+0.96*pow(1.0-cosv,5.0);
  // light that entered the pane from the lit water travels inside it to the edge: follows the caustic floor next to this point
  let fp=clamp(p.xz,vec2f(-0.43),vec2f(0.43)); let o=0.05;   // nearest floor point inside the tank, 5-tap blur
  let lloc=edgeLum(floorShade(fp))*0.4+(edgeLum(floorShade(fp+vec2f(o,0.0)))+edgeLum(floorShade(fp-vec2f(o,0.0)))+edgeLum(floorShade(fp+vec2f(0.0,o)))+edgeLum(floorShade(fp-vec2f(0.0,o))))*0.15;
  // plus the lit floor seen down along the pane (the view, mirrored into the tank by the pane, continued steeply down): varies along the edge and with the view
  let po=normalize(i.uv+vec2f(1e-6))*sign(dot(p.xz,i.uv)+1e-6); var hv=rd.xz; hv-=2.0*max(dot(hv,po),0.0)*po;
  let dv=normalize(vec3f(hv.x,-abs(rd.y)-0.4,hv.y)); let fv=clamp(p.xz+dv.xz*(p.y+D)/(-dv.y),vec2f(-0.45),vec2f(0.45));
  let lit=0.35*lloc+0.65*edgeLum(floorShade(fv));
  let hgt=max(p.y+D,0.0);
  let sl=normalize(u.sun.xyz); let pnv=select(vec3f(i.uv.x,0.0,i.uv.y),vec3f(0.0,1.0,0.0),dot(i.uv,i.uv)<0.25);
  let key=EDGE_KEY*abs(dot(pnv,sl));
  let glow=EDGE_PIPE*lit*exp(-hgt*EDGE_FALL)+EDGE_AMB*(0.3+u.b.x)*(0.5+exp(-hgt*2.0))+key;
  let pipe=EDGE_COL*(glow*(0.3+0.7*cosv)*(1.0-F)+EDGE_MIN*(1.0-cosv));   // brightest looking straight into the edge; EDGE_MIN only at grazing (the top rim at low pitch never goes black), not a fill
  let rr=reflect(rd,n);
  let refl=env(rr,GLASS_TOP)*F*mix(vec3f(1.0),EDGE_COL*1.25,0.4);
  let spec=EDGE_SPEC*pow(max(dot(rr,normalize(u.sun.xyz)),0.0),60.0)*mix(LIGHT_COL,EDGE_COL,0.5);
  let col=EDGE_COL*0.003+pipe+refl+spec;
  return vec4f(col,1.0);
}
`;

// Background gradient (separate entry so it can run before the scene without needing the glass/water code paths)
export const BG_WGSL = COMMON + /* wgsl */`
@vertex fn vs(@builtin(vertex_index) vi:u32) -> @builtin(position) vec4f {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[vi], 0.9999, 1.0);
}
@fragment fn fs(@builtin(position) fp:vec4f) -> @location(0) vec4f {
  let res = vec2f(u.b.z, u.b.w);
  let ndc = vec2f(fp.x/res.x*2.0-1.0, 1.0-fp.y/res.y*2.0);
  let rd = normalize(u.fwd.xyz + u.rgt.xyz*ndc.x*u.fwd.w*u.rgt.w + u.upv.xyz*ndc.y*u.fwd.w);
  return vec4f(bgCol(rd, ndc),1.0);
}
`;

export const RIPPLE_WGSL = (N) => /* wgsl */`
const N:u32=${N}u;
struct R { t:f32, amp:f32, ambient:f32, pad:f32, clk:array<vec4f,8> }  // clk: (u, v, startTime, active)
@group(0) @binding(0) var<uniform> r:R;
@group(0) @binding(1) var<storage,read> hin:array<f32>;
@group(0) @binding(2) var<storage,read_write> hout:array<f32>;
@compute @workgroup_size(8,8) fn addRipple(@builtin(global_invocation_id) g:vec3u) {
  let i=g.y*N+g.x; let p=vec2f(f32(g.x),f32(g.y))/f32(N-1u);
  var w=0.0;
  if (r.amp > 0.0) {
    for (var j=0; j<8; j++) {
      let c=r.clk[j];
      if (c.w > 0.5) {
        let age=r.t-c.z; let d=length(p-c.xy); let front=0.20*age;
        let e=exp(-pow((d-front)/0.075,2.0));
        w += sin((d-front)*2.0*3.14159/0.085)*e*exp(-age*0.55)/(1.0+d*5.0);
        w -= 1.2*exp(-d*d/0.0007)*exp(-age*7.0);
      }
    }
    if (r.ambient > 0.5) {
      let t=r.t;
      w += 0.30*sin(dot(p,vec2f(0.8,0.6))*2.0*3.14159*7.0 - t*1.5);
      w += 0.25*sin(dot(p,vec2f(-0.5,0.85))*2.0*3.14159*9.5 - t*1.9 + 1.3);
      w += 0.20*sin(dot(p,vec2f(0.95,-0.3))*2.0*3.14159*12.0 - t*2.4 + 2.1);
    }
  }
  hout[i]=hin[i]+r.amp*w;
}
`;

// Bloom (linear HDR, before the tone map): soft-knee prefilter + 13-tap downsample chain (1/2 .. 1/64), then tent upsample added back up the chain.
// The final pass adds BLOOM_STR x the 1/2-res level. Karis-weighted first downsample keeps single bright glint pixels from flickering.
export const BLOOM_WGSL = /* wgsl */`
const BLOOM_THRESH : f32 = 0.9;   // linear scene radiance (pre-exposure) where the bloom starts (display ~0.9 at EXPOSURE 2): dark water / haze does not bloom
const BLOOM_KNEE   : f32 = 0.6;   // soft-knee width around the threshold
const BLOOM_CLAMP  : f32 = 40.0;  // prefilter cap (sun-glint fireflies)
const BLOOM_RADIUS : f32 = 1.0;   // upsample tent radius in source texels (spread per level)
@group(0) @binding(0) var src:texture_2d<f32>;
@group(0) @binding(1) var smp:sampler;
struct VO { @builtin(position) pos:vec4f, @location(0) uv:vec2f }
@vertex fn vs(@builtin(vertex_index) vi:u32) -> VO {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  var o:VO; o.pos=vec4f(p[vi],0.0,1.0); o.uv=vec2f(p[vi].x*0.5+0.5, 0.5-p[vi].y*0.5); return o;
}
fn S(uv:vec2f)->vec3f { return textureSampleLevel(src,smp,uv,0.0).rgb; }
fn luma(c:vec3f)->f32 { return dot(c, vec3f(0.2126,0.7152,0.0722)); }
fn knee(c:vec3f)->vec3f {
  let c2=min(c,vec3f(BLOOM_CLAMP)); let br=max(c2.r,max(c2.g,c2.b));
  var rq=clamp(br-BLOOM_THRESH+BLOOM_KNEE,0.0,2.0*BLOOM_KNEE); rq=rq*rq/(4.0*BLOOM_KNEE+1e-5);
  return c2*max(rq,br-BLOOM_THRESH)/max(br,1e-5);
}
// 13-tap (Jimenez 2014): 4 overlapping 2x2 boxes + centre box; first=true -> soft knee + Karis weights per box
fn down13(uv:vec2f, first:bool)->vec3f {
  let t=1.0/vec2f(textureDimensions(src));
  let a=S(uv+t*vec2f(-2.0,-2.0)); let b=S(uv+t*vec2f(0.0,-2.0)); let c=S(uv+t*vec2f(2.0,-2.0));
  let d=S(uv+t*vec2f(-1.0,-1.0)); let e=S(uv+t*vec2f(1.0,-1.0));
  let f=S(uv+t*vec2f(-2.0,0.0)); let g=S(uv); let h=S(uv+t*vec2f(2.0,0.0));
  let i=S(uv+t*vec2f(-1.0,1.0)); let j=S(uv+t*vec2f(1.0,1.0));
  let k=S(uv+t*vec2f(-2.0,2.0)); let l=S(uv+t*vec2f(0.0,2.0)); let m=S(uv+t*vec2f(2.0,2.0));
  var g0=(d+e+i+j)*0.25; var g1=(a+b+f+g)*0.25; var g2=(b+c+g+h)*0.25; var g3=(f+g+k+l)*0.25; var g4=(g+h+l+m)*0.25;
  if (!first) { return g0*0.5+(g1+g2+g3+g4)*0.125; }
  g0=knee(g0); g1=knee(g1); g2=knee(g2); g3=knee(g3); g4=knee(g4);
  let w0=0.5/(1.0+luma(g0)); let w1=0.125/(1.0+luma(g1)); let w2=0.125/(1.0+luma(g2)); let w3=0.125/(1.0+luma(g3)); let w4=0.125/(1.0+luma(g4));
  return (g0*w0+g1*w1+g2*w2+g3*w3+g4*w4)/(w0+w1+w2+w3+w4);
}
@fragment fn fsDownFirst(i:VO) -> @location(0) vec4f { return vec4f(down13(i.uv,true),1.0); }
@fragment fn fsDown(i:VO) -> @location(0) vec4f { return vec4f(down13(i.uv,false),1.0); }
// 3x3 tent of the coarser level, added (blend one+one) onto this level
@fragment fn fsUp(i:VO) -> @location(0) vec4f {
  let t=BLOOM_RADIUS/vec2f(textureDimensions(src)); let uv=i.uv;
  var c=S(uv)*4.0;
  c+=(S(uv+vec2f(t.x,0.0))+S(uv-vec2f(t.x,0.0))+S(uv+vec2f(0.0,t.y))+S(uv-vec2f(0.0,t.y)))*2.0;
  c+=S(uv+t)+S(uv-t)+S(uv+vec2f(t.x,-t.y))+S(uv+vec2f(-t.x,t.y));
  return vec4f(c/16.0,1.0);
}
`;

// Final pass: the scene is rendered + blended LINEAR into an rgba16float target; bloom is added, then exposure + ACES + gamma happen once, here.
const FINAL_COMMON = /* wgsl */`
const EXPOSURE : f32 = 2.0;                        // linear gain before the curve
const GRADE_TINT : vec3f = vec3f(0.98, 1.0, 1.02); // linear teal-leaning white balance (pre-curve)
const GRADE_SAT : f32 = 1.15;                      // saturation boost after the curve (1 = none)
const GRADE_MID : f32 = 1.38;                      // >1 darkens mid-tones (more contrast between dark water and bright caustics)
const BLOOM_STR : f32 = 0.06;                      // bloom mix (the chain sums 6 levels, so ~0.36 x the knee-filtered energy is spread around)
const DITHER    : f32 = 1.0;                       // output dither amplitude in 8-bit LSBs (triangular PDF): removes banding in the dark gradients
@group(0) @binding(0) var hdr:texture_2d<f32>;
@group(0) @binding(1) var bloom:texture_2d<f32>;
@group(0) @binding(2) var bsm:sampler;
@vertex fn vs(@builtin(vertex_index) vi:u32) -> @builtin(position) vec4f {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[vi], 0.0, 1.0);
}
fn aces(x:vec3f)->vec3f { return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), vec3f(0.0), vec3f(1.0)); }
fn toneMapped(fp:vec4f)->vec3f {
  let uv = fp.xy/vec2f(textureDimensions(hdr));
  let lin = max(textureLoad(hdr, vec2i(fp.xy), 0).rgb, vec3f(0.0)) + BLOOM_STR*textureSampleLevel(bloom, bsm, uv, 0.0).rgb;
  var c = aces(lin*EXPOSURE*GRADE_TINT);
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = clamp(mix(vec3f(l), c, GRADE_SAT), vec3f(0.0), vec3f(1.0));
  return pow(c, vec3f(GRADE_MID));
}
// triangular noise in [-1,1] LSB (two decorrelated interleaved-gradient samples)
fn ign2(p:vec2f)->f32 { return fract(52.9829189*fract(dot(p,vec2f(0.06711056,0.00583715)))); }
fn dither(fp:vec4f)->f32 { return (ign2(fp.xy) + ign2(fp.xy + vec2f(47.0,17.0)) - 1.0)*DITHER/255.0; }
fn toSrgb(c:vec3f)->vec3f { return select(1.055*pow(c,vec3f(1.0/2.4))-0.055, c*12.92, c<=vec3f(0.0031308)); }
fn fromSrgb(c:vec3f)->vec3f { return select(pow((c+0.055)/1.055,vec3f(2.4)), c/12.92, c<=vec3f(0.04045)); }
`;
export const FINAL_WGSL = FINAL_COMMON + /* wgsl */`
@fragment fn fs(@builtin(position) fp:vec4f) -> @location(0) vec4f { return vec4f(pow(toneMapped(fp), vec3f(1.0/2.2)) + dither(fp), 1.0); }
// *-srgb swapchain: hardware encodes, so dither in the encoded domain and hand back linear
@fragment fn fsSrgb(@builtin(position) fp:vec4f) -> @location(0) vec4f { return vec4f(fromSrgb(clamp(pow(toneMapped(fp), vec3f(1.0/2.2)) + dither(fp), vec3f(0.0), vec3f(1.0))), 1.0); }   // the hardware sRGB encode undoes fromSrgb: stored bytes = fs (gamma 2.2), not a different curve
`;
