// Damped wave dynamics on the deviation e = h - h*  (periodic N x N grid, h = 1/N, domain = 1 x 1 tank footprint).
//   e_tt = c^2 Lap(e) - kappa e - gamma e_t + nu Lap(e_t) - s Lap(Lap(e))
// Scheme: semi-implicit (symplectic) Euler:  v' = v + dt*a(e, v);  e' = e + dt*v'   (ping-pong buffers, 5-pt Laplacian, 13-pt biharmonic)
export const RELAX_WGSL = (N) => /* wgsl */`
const N:u32=${N}u; const NI:i32=${N};
struct P { dt:f32, c2:f32, kappa:f32, gamma:f32, nu:f32, s:f32, ih2:f32, p0:f32 }
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> eIn:array<f32>;
@group(0) @binding(2) var<storage,read> vIn:array<f32>;
@group(0) @binding(3) var<storage,read_write> eOut:array<f32>;
@group(0) @binding(4) var<storage,read_write> vOut:array<f32>;
@group(0) @binding(5) var<storage,read> hs:array<f32>;
@group(0) @binding(6) var<storage,read> hp:array<f32>;
@group(0) @binding(7) var<storage,read_write> hT:array<f32>;
@group(0) @binding(8) var<storage,read_write> stats:array<f32>;
// continuous ambient forcing: 12 swell + 20 fine-ripple travelling waves, each vec4(2*pi*kx, 2*pi*ky, phase(t), amplitude); amplitude 0 = unused
@group(0) @binding(9) var<uniform> amb:array<vec4f,32>;
fn ix(x:i32,y:i32)->u32 { return u32(((y&(NI-1))*NI)+(x&(NI-1))); }
@compute @workgroup_size(8,8) fn step(@builtin(global_invocation_id) g:vec3u) {
  let x=i32(g.x); let y=i32(g.y); let i=ix(x,y);
  let e0=eIn[i];
  let en=eIn[ix(x+1,y)]+eIn[ix(x-1,y)]+eIn[ix(x,y+1)]+eIn[ix(x,y-1)];
  let ed=eIn[ix(x+1,y+1)]+eIn[ix(x-1,y+1)]+eIn[ix(x+1,y-1)]+eIn[ix(x-1,y-1)];
  let e2=eIn[ix(x+2,y)]+eIn[ix(x-2,y)]+eIn[ix(x,y+2)]+eIn[ix(x,y-2)];
  let lapE=(en-4.0*e0)*p.ih2;
  let bi=(20.0*e0-8.0*en+2.0*ed+e2)*p.ih2*p.ih2;
  let v0=vIn[i];
  let vn=vIn[ix(x+1,y)]+vIn[ix(x-1,y)]+vIn[ix(x,y+1)]+vIn[ix(x,y-1)];
  let lapV=(vn-4.0*v0)*p.ih2;
  var F=0.0; let px=f32(x)/f32(N); let py=f32(y)/f32(N);
  for (var m=0; m<32; m++) { let q=amb[m]; if (q.w!=0.0) { F+=q.w*cos(q.x*px+q.y*py-q.z); } }
  let a=p.c2*lapE-p.kappa*e0-p.s*bi-p.gamma*v0+p.nu*lapV+F;
  let v1=v0+p.dt*a;
  vOut[i]=v1; eOut[i]=e0+p.dt*v1;
}
@compute @workgroup_size(8,8) fn compose(@builtin(global_invocation_id) g:vec3u) { let i=g.y*N+g.x; hT[i]=hs[i]+eIn[i]; }
// new target: keep the surface, e += h*_old - h*_new   (in place on eOut)
@compute @workgroup_size(8,8) fn retarget(@builtin(global_invocation_id) g:vec3u) { let i=g.y*N+g.x; eOut[i]=eOut[i]+hp[i]-hs[i]; }
@compute @workgroup_size(8,8) fn addKick(@builtin(global_invocation_id) g:vec3u) { let i=g.y*N+g.x; vOut[i]=vOut[i]+hp[i]; }
var<workgroup> r0:array<f32,256>; var<workgroup> r1:array<f32,256>; var<workgroup> r2:array<f32,256>; var<workgroup> r3:array<f32,256>;
// stats = [mean e^2, mean h*^2, max|e|, non-finite count]
@compute @workgroup_size(256) fn reduce(@builtin(local_invocation_index) lid:u32) {
  var a0=0.0; var a1=0.0; var a2=0.0; var a3=0.0;
  for (var i=lid; i<N*N; i+=256u) {
    let e=eIn[i]; let h=hs[i];
    if (abs(e)<1.0e30) { a0+=e*e; a2=max(a2,abs(e)); } else { a3+=1.0; }
    if (abs(h)<1.0e30) { a1+=h*h; }
  }
  r0[lid]=a0; r1[lid]=a1; r2[lid]=a2; r3[lid]=a3; workgroupBarrier();
  for (var s=128u; s>0u; s=s>>1u) {
    if (lid<s) { r0[lid]+=r0[lid+s]; r1[lid]+=r1[lid+s]; r2[lid]=max(r2[lid],r2[lid+s]); r3[lid]+=r3[lid+s]; }
    workgroupBarrier();
  }
  if (lid==0u) { let n=f32(N*N); stats[0]=r0[0]/n; stats[1]=r1[0]/n; stats[2]=r2[0]; stats[3]=r3[0]; }
}
`;

export function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

// Band-limited random wave field + rings + standing waves (periodic). Returns {e, v} normalised so rms(e) = rmsTarget.
// Plane waves are travelling (v = omega*A*sin(theta)); ring pulses travel outward (v = -c f'(r)); standing waves have v = 0.
export function genIC(N, seed, rmsTarget, P) {
  const rnd = mulberry32(seed), TP = Math.PI * 2, e = new Float32Array(N * N), v = new Float32Array(N * N);
  const omega = kk => Math.sqrt(P.c2 * (TP * kk) ** 2 + P.kappa + P.s * (TP * kk) ** 4);
  const tabs = (k, ph) => { const a = new Float32Array(N), b = new Float32Array(N); for (let i = 0; i < N; i++) { a[i] = Math.cos(TP * k * i / N); b[i] = Math.sin(TP * k * i / N); } return [a, b]; };
  const kmax = Math.max(2.5, P.kmax), M = 36;
  for (let m = 0; m < M; m++) {
    let kx, ky, kk; do { kx = Math.round((rnd() * 2 - 1) * kmax); ky = Math.round((rnd() * 2 - 1) * kmax); kk = Math.hypot(kx, ky); } while (kk < 1.5 || kk > kmax);
    const A = 1 / (kk * kk), ph = rnd() * TP, w = omega(kk), [cx, sx] = tabs(kx), [cy, sy] = tabs(ky), cp = Math.cos(ph), sp = Math.sin(ph);
    for (let j = 0; j < N; j++) { const c1 = cy[j], s1 = sy[j]; for (let i = 0; i < N; i++) {
      // theta = a + b + ph; cos(theta)=cos(a+b)cp - sin(a+b)sp
      const cab = cx[i] * c1 - sx[i] * s1, sab = sx[i] * c1 + cx[i] * s1, id = j * N + i;
      e[id] += A * (cab * cp - sab * sp); v[id] += w * A * (sab * cp + cab * sp);
    } }
  }
  const rms0 = (() => { let s = 0; for (let i = 0; i < e.length; i++) s += e[i] * e[i]; return Math.sqrt(s / e.length); })();
  const gA = 0.8 * rms0;   // rings / standing waves carry comparable amplitude
  for (let r = 0; r < 3; r++) {
    const cx0 = rnd(), cy0 = rnd(), r0 = 0.08 + rnd() * 0.22, wd = 0.06 + rnd() * 0.04, lam = 0.14 + rnd() * 0.08, A = gA * 1.0;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      let dx = i / N - cx0, dy = j / N - cy0; dx -= Math.round(dx); dy -= Math.round(dy);
      const rr = Math.hypot(dx, dy), q = (rr - r0) / wd, env = Math.exp(-q * q), ph = TP * (rr - r0) / lam;
      const f = env * Math.cos(ph), fp = env * (-2 * q / wd * Math.cos(ph) - TP / lam * Math.sin(ph));
      e[j * N + i] += A * f; v[j * N + i] += -Math.sqrt(P.c2) * A * fp;
    }
  }
  for (let r = 0; r < 2; r++) {
    const kx = 2 + Math.floor(rnd() * 4), ky = 2 + Math.floor(rnd() * 4), A = gA, ph1 = rnd() * TP, ph2 = rnd() * TP;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) e[j * N + i] += A * Math.cos(TP * kx * i / N + ph1) * Math.cos(TP * ky * j / N + ph2);
  }
  let s = 0, mean = 0, mv = 0; for (let i = 0; i < e.length; i++) { mean += e[i]; mv += v[i]; } mean /= e.length; mv /= e.length;
  for (let i = 0; i < v.length; i++) v[i] -= mv;   // zero-mean velocity (k=0 mode would otherwise drift when kappa=0)
  for (let i = 0; i < e.length; i++) { e[i] -= mean; s += e[i] * e[i]; }
  const f = rmsTarget / (Math.sqrt(s / e.length) || 1);
  for (let i = 0; i < e.length; i++) { e[i] *= f; v[i] *= f; }
  return { e, v };
}
