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
const FLOOR_GAIN     : f32 = 1.9;                        // brightness of the lit caustic on the floor
const LIGHT_COL      : vec3f = vec3f(1.0, 0.95, 0.86);
const BOX_GAIN       : f32 = 6.0;                        // overhead soft box (reflected in water)
const STRIP_GAIN     : f32 = 4.0;                        // side / back strip lights
const STRIP_WARM     : vec3f = vec3f(0.877, 0.778, 0.675);  // warm side strip colour (old 1,0.78,0.55 pulled ~55% toward neutral: the glass reflection no longer reads cream/orange)
const F0_WATER       : f32 = 0.02;
const SUN_POW        : f32 = 900.0;
const SUN_INT        : f32 = 18.0;
const GLASS_REFL     : f32 = 0.25;                       // glass wall reflection colour strength (linear)
const GLASS_ALPHA    : f32 = 1.1;                        // glass wall opacity from the Fresnel term (how much it hides what is behind)
const GLASS_EDGE_W   : f32 = 0.006;                       // bright edge half width (world units)
const WALL_TOP       : f32 = 0.15;                        // glass wall height above water level
const TABLE_GLOSS    : f32 = 0.2;
const SPILL_GAIN     : f32 = 0.09;                        // light spilled on table around tank
const PHASE_KNEE     : f32 = 1.0;                        // god-ray phase factor: unchanged below, soft-limited above (HG forward peak at g=0.5 is ~4.5x -> blows out when looking into the light)
const PHASE_MAX      : f32 = 1.8;
const RAY_FLOOR      : f32 = 0.6;                        // god rays: volume level subtracted before scattering (removes the uniform fog, keeps the shaft contrast)
const RAY_REGAIN     : f32 = 1.3;                        // gain applied after the floor subtraction (restores shaft peak strength)
const VIEW_SLOPE     : f32 = 0.3;                       // LOOK ONLY: scales surface slopes used for refraction/reflection shading (1 = physical)
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
  if (which==2 && u.m.z>0.5) { q=floor(q+vec2f(0.5)); }   // 'nearest' caustic sampling: chunky texels
  let f=floor(q); let a=q-f;
  let x0=min(i32(f.x),n-1); let y0=min(i32(f.y),n-1); let x1=min(x0+1,n-1); let y1=min(y0+1,n-1);
  var v00=0.0; var v10=0.0; var v01=0.0; var v11=0.0;
  if (which==0) { v00=hxb[y0*n+x0]; v10=hxb[y0*n+x1]; v01=hxb[y1*n+x0]; v11=hxb[y1*n+x1]; }
  else if (which==1) { v00=hyb[y0*n+x0]; v10=hyb[y0*n+x1]; v01=hyb[y1*n+x0]; v11=hyb[y1*n+x1]; }
  else { v00=Cb[y0*n+x0]; v10=Cb[y0*n+x1]; v01=Cb[y1*n+x0]; v11=Cb[y1*n+x1]; }
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
  let S=max(1,i32(u.g2.x)); let ds=tEnd/f32(S); let sig=u.g.z; let ga=u.g.w;
  let Lw=u.lw.xyz; let cs=-dot(Lw,r);   // light travel dir inside the water (refracted); cos(scatter angle) = dot(L, -r)
  let hg=(1.0-ga*ga)/pow(max(1.0+ga*ga-2.0*ga*cs,1e-3),1.5);
  var phase=0.3+0.7*hg;
  if (phase>PHASE_KNEE) { phase=PHASE_KNEE+(PHASE_MAX-PHASE_KNEE)*tanh((phase-PHASE_KNEE)/(PHASE_MAX-PHASE_KNEE)); }
  let j=ign(px);
  // transmittance (view path + light path) advanced multiplicatively: both are exp of a linear function of s
  let ilz=1.0/max(-Lw.y,0.2);   // light path length per unit depth
  let kv=ABSORB*0.5+vec3f(sig)+ABSORB*0.6*ilz*(-r.y);
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


// procedural dark-room environment: soft boxes / strips on a near-black gradient
fn env(d:vec3f)->vec3f {
  var c = mix(vec3f(0.003,0.004,0.006), vec3f(0.018,0.026,0.045), clamp(d.y*0.5+0.5,0.0,1.0));
  let dk = keyRot(d);
  if (dk.y > 0.12) {
    let q = dk.xz / dk.y;
    let b = smoothstep(0.15, 0.0, max(abs(q.x - 0.1) - 0.85, abs(q.y + 0.15) - 0.5));
    c += BOX_GAIN * b * vec3f(1.0, 0.96, 0.9);
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
  let c = bil(2, p + 0.5);
  let l = pow(1.0 - exp(-max(c-0.4,0.0) * 1.0 * u.a.z), 1.6);
  let edge = smoothstep(0.0, 0.16, 0.5 - max(abs(p.x), abs(p.y)));
  let vig = mix(0.5, 1.0, edge);
  let glow = LIGHT_COL * u.c.x * 0.03 * exp(-dot(p,p)*5.0);
  return FLOOR_ALBEDO * (FLOOR_AMBIENT + LIGHT_COL * l * FLOOR_GAIN + glow) * vig;
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
  var gx=bil(0,i.uv)*R*VIEW_SLOPE; var gz=bil(1,i.uv)*R*VIEW_SLOPE;
  if (u.m.y>0.5) { let fo=select(-fnr,fnr,fnr.y>0.0); let fy=max(abs(fo.y),1e-9); gx=-fo.x/fy*VIEW_SLOPE; gz=-fo.z/fy*VIEW_SLOPE; }
  let n=normalize(vec3f(-gx, 1.0, -gz));
  let P=i.w; let rd=normalize(P-u.eye.xyz);
  let cosi=-dot(rd,n);
  if (cosi < 0.0) { // seen from below the surface: translucent bluish mirror
    let rf=reflect(rd,-n);
    let col=waterTintAt(0.2)*3.0 + vec3f(0.02,0.05,0.07);
    return vec4f(col*0.55, 0.55);
  }
  let F=F0_WATER+(1.0-F0_WATER)*pow(1.0-cosi,5.0);
  let r=refract(rd,n,1.0/IOR);
  var col=vec3f(0.0);
  let t=(-D-P.y)/min(r.y,-1e-3);
  let Fh=P+r*t;
  if (abs(Fh.x)<=0.5 && abs(Fh.z)<=0.5) {
    col=floorShade(Fh.xz)*exp(-ABSORB*t);
    col+=waterTintAt(0.0)*(1.0-exp(-t*0.9));
    col+=godRays(P,r,t,i.pos.xy);
  } else { // ray leaves through a side wall: shade the inner wall
    let tx=select(1e9,(sign(r.x)*0.5-P.x)/r.x,abs(r.x)>1e-5);
    let tz=select(1e9,(sign(r.z)*0.5-P.z)/r.z,abs(r.z)>1e-5);
    let tw=min(tx,tz); let hw=P+r*tw;
    let hs=bilH(clamp(hw.xz+vec2f(0.5),vec2f(0.0),vec2f(1.0)))*R;
    let df=clamp((hs-hw.y)/D,0.0,1.0);   // depth below the LOCAL surface at the wall point
    col=waterTintAt(df)*1.2*exp(-ABSORB*tw*0.5)+vec3f(0.004,0.010,0.012);
    col+=godRays(P,r,tw,i.pos.xy);
  }
  let rr=reflect(rd,n);
  var refl=env(rr);
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
  var refl=env(rr)*0.5+env(normalize(rr+vec3f(0.06,0.0,0.0)))*0.25+env(normalize(rr-vec3f(0.06,0.0,0.0)))*0.25;
  var col=vec3f(0.006,0.0065,0.008)+refl*F*TABLE_GLOSS*0.35;
  let spill=exp(-d*5.0)*SPILL_GAIN*(0.2+u.b.x);
  col+=spill*vec3f(0.30,0.55,0.62);
  col+=exp(-d*28.0)*0.08*vec3f(0.6,0.85,0.9)*(0.2+u.b.x);
  col*=exp(-length(p)*0.22);
  // dark contact shadow right at the glass base
  col*=mix(0.35,1.0,smoothstep(0.0,0.02,d));
  return vec4f(col,1.0);
}

// ---------- glass walls ----------
@vertex fn vsGlass(@builtin(vertex_index) vi:u32, @builtin(instance_index) ii:u32) -> FS {
  var cx=array<f32,6>(0.0,1.0,0.0,1.0,1.0,0.0); var cy=array<f32,6>(0.0,0.0,1.0,0.0,1.0,1.0);
  let s=cx[vi]-0.5; let y=mix(-u.a.x, WALL_TOP, cy[vi]);
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
  let refl=env(reflect(rd,n))*F*GLASS_REFL;
  let dTop=WALL_TOP-i.lc.y; let dBot=i.lc.y+u.a.x; let dSide=0.5-abs(i.lc.x);
  let dE=min(dTop,min(dBot,dSide));
  let edge=smoothstep(GLASS_EDGE_W,0.0,dE);
  // local waterline: the water mesh height at this wall point (wall uv: ii 0/1 -> z=+/-0.5 (uv.y=1/0), ii 2/3 -> x=+/-0.5 (uv.x=1/0); the other uv = lc.x+0.5)
  let alongX=(ii<2u); let fixedC=select(0.0,1.0,(ii==0u)||(ii==2u));
  let hw=wallH(i.lc.x+0.5,fixedC,alongX);
  let under=1.0-smoothstep(hw-0.003,hw+0.003,i.lc.y);
  let wa=under*0.30;
  let inside=waterTintAt(clamp((hw-i.lc.y)/u.a.x,0.0,1.0))*(2.2)*wa;
  let men=exp(-pow((i.lc.y-hw)/0.004,2.0));   // faint meniscus line on the waterline
  // faint vertical sheen band
  let sheen=smoothstep(0.35,0.0,abs(i.lc.x-0.18))*0.02*F;
  var rgb=refl+inside+vec3f(0.55,0.75,0.85)*edge*0.7+sheen+vec3f(0.35,0.55,0.62)*men*0.35;
  let a=clamp(F*GLASS_ALPHA+wa+edge*0.7+men*0.25,0.0,1.0);
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
  var c = vec3f(0.0025,0.003,0.0045)*(1.0+4.0*pow(1.0-abs(rd.y),4.0));
  c += vec3f(0.004,0.007,0.009)*u.b.x*exp(-length(ndc)*1.6);
  c *= 1.0-0.45*dot(ndc*0.6,ndc*0.6);
  let sdot=dot(rd,normalize(u.sun.xyz));  // small sun disc (visible from side views)
  c += LIGHT_COL*6.0*smoothstep(0.9990,0.9996,sdot)+LIGHT_COL*0.05*pow(max(sdot,0.0),64.0);
  return vec4f(c,1.0);
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

// Final pass: the scene is rendered + blended LINEAR into an rgba16float target; exposure + ACES + gamma happen once, here.
const FINAL_COMMON = /* wgsl */`
const EXPOSURE : f32 = 2.0;                        // linear gain before the curve
const GRADE_TINT : vec3f = vec3f(0.98, 1.0, 1.02); // linear teal-leaning white balance (pre-curve)
const GRADE_SAT : f32 = 1.15;                      // saturation boost after the curve (1 = none)
const GRADE_MID : f32 = 1.38;                      // >1 darkens mid-tones (more contrast between dark water and bright caustics)
@group(0) @binding(0) var hdr:texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) vi:u32) -> @builtin(position) vec4f {
  let p = array<vec2f,3>(vec2f(-1.0,-1.0), vec2f(3.0,-1.0), vec2f(-1.0,3.0));
  return vec4f(p[vi], 0.0, 1.0);
}
fn aces(x:vec3f)->vec3f { return clamp((x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14), vec3f(0.0), vec3f(1.0)); }
fn toneMapped(fp:vec4f)->vec3f {
  var c = aces(max(textureLoad(hdr, vec2i(fp.xy), 0).rgb, vec3f(0.0))*EXPOSURE*GRADE_TINT);
  let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = clamp(mix(vec3f(l), c, GRADE_SAT), vec3f(0.0), vec3f(1.0));
  return pow(c, vec3f(GRADE_MID));
}
`;
export const FINAL_WGSL = FINAL_COMMON + /* wgsl */`
@fragment fn fs(@builtin(position) fp:vec4f) -> @location(0) vec4f { return vec4f(pow(toneMapped(fp), vec3f(1.0/2.2)), 1.0); }
@fragment fn fsSrgb(@builtin(position) fp:vec4f) -> @location(0) vec4f { return vec4f(toneMapped(fp), 1.0); }   // *-srgb swapchain: hardware encodes
`;
