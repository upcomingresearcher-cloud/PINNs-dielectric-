"""Phase 2: symbolic verification of reference solutions used in the protocol.
Writes results/validation/phase2_derivation_checks.txt. Every check must print PASS."""
import sympy as sp, pathlib
out=[]
def check(name, expr):
    """Record PASS/FAIL for a symbolic identity that should simplify to zero; return whether it passed."""
    ok = sp.simplify(expr) == 0
    out.append(f"{'PASS' if ok else 'FAIL'}  {name}"); return ok
x,a,k,V,L = sp.symbols('x a kappa V L', positive=True)
# ---- P1 layered capacitor (eps1=1 for x<a, eps2=kappa for x>a), phi(0)=1, phi(1)=0
D = 1/(a+(1-a)/k)
p1 = 1 - D*x; p2 = D*(1-x)/k
check("P1 BC phi1(0)=1", p1.subs(x,0)-1)
check("P1 BC phi2(1)=0", p2.subs(x,1))
check("P1 potential continuity at a", (p1-p2).subs(x,a))
check("P1 flux continuity eps1*phi1'=kappa*phi2' at a", (sp.diff(p1,x)-k*sp.diff(p2,x)).subs(x,a))
check("P1 PDE in each material (phi''=0)", sp.diff(p1,x,2)+sp.diff(p2,x,2))
check("P1 E=-phi' gives D=eps*E=D_exact in mat1", -sp.diff(p1,x)-D)
check("P1 D in mat2 equals D_exact", -k*sp.diff(p2,x)-D)
W = sp.Rational(1,2)*(sp.integrate(sp.diff(p1,x)**2,(x,0,a))+k*sp.integrate(sp.diff(p2,x)**2,(x,a,1)))
check("P1 energy capacitance 2W/V^2 equals flux capacitance D (V=1)", 2*W-D)
# harmonic-mean flux scale used by F6 equals D_exact when a=1/2 (documented coincidence)
eh = 2*k/(1+k)
check("F6 scale eps_h*V/L equals D_exact for a=1/2", eh - D.subs(a,sp.Rational(1,2)))
# wrong uniform-permittivity solution vs kappa=80: relative E error in material 1
Dk = D.subs({a:sp.Rational(1,2),k:80})
out.append(f"INFO  kappa=80: D_exact={float(Dk):.6f}; uniform-eps E1=1 -> rel. error {float(abs(1-Dk)/Dk):.4f}")
# ---- F4-full buffer system (1D): solvable for all kappa>0, 0<a<1
c10,c11,c20,c21,e1,e2 = sp.symbols('c10 c11 c20 c21 e1 e2')
Mtx = sp.Matrix([[1,0,0,0],[0,0,1,1],[1,a,-1,-a],[0,e1,0,-e2]])
check("F4-full buffer determinant = +(a*e2+(1-a)*e1) > 0 (hand derivation had sign error, corrected)", Mtx.det() - (a*e2+(1-a)*e1))
# ---- P2 dielectric cylinder in uniform field E0 (x-direction), radius R, k = eps_in/eps_out
r,th,R,E0,ei,eo = sp.symbols('r theta R E0 eps_in eps_out', positive=True)
K = (ei-eo)/(ei+eo)
pin  = -E0*2*eo/(ei+eo)*r*sp.cos(th)
pout = -E0*r*sp.cos(th) + E0*K*R**2*sp.cos(th)/r
lap = lambda f: sp.diff(r*sp.diff(f,r),r)/r + sp.diff(f,th,2)/r**2
check("P2 Laplace inside", lap(pin)); check("P2 Laplace outside", lap(pout))
check("P2 potential continuity at r=R", (pin-pout).subs(r,R))
check("P2 normal-D continuity at r=R", (ei*sp.diff(pin,r)-eo*sp.diff(pout,r)).subs(r,R))
check("P2 interior field magnitude 2*eps_out/(eps_in+eps_out)*E0", -sp.diff(pin,r)/sp.cos(th) - 2*eo/(ei+eo)*E0)
# ---- P3 2D line charge lam at (0,h) in eps1 (y>0); eps2 (y<0); interface y=0
X,Y,h,lam,e1_,e2_ = sp.symbols('X Y h lambda eps1 eps2', positive=True)
r1 = sp.sqrt(X**2+(Y-h)**2); r2 = sp.sqrt(X**2+(Y+h)**2)
up = -lam/(2*sp.pi*e1_)*(sp.log(r1) + (e1_-e2_)/(e1_+e2_)*sp.log(r2))
dn = -lam/(2*sp.pi)*2/(e1_+e2_)*sp.log(r1)
lap2 = lambda f: sp.diff(f,X,2)+sp.diff(f,Y,2)
check("P3 harmonic above (away from source)", lap2(up)); check("P3 harmonic below", lap2(dn))
check("P3 potential continuity on y=0", (up-dn).subs(Y,0))
check("P3 normal-D continuity on y=0", (e1_*sp.diff(up,Y)-e2_*sp.diff(dn,Y)).subs(Y,0))
# source strength: flux of D=-eps1 grad(phi_free) through a small circle = lam (2D Gauss law)
check("P3 free-space term encloses charge lam", sp.integrate(-e1_*sp.diff(-lam/(2*sp.pi*e1_)*sp.log(r),r)*r,(th,0,2*sp.pi)) - lam)
txt="\n".join(out); print(txt)
p=pathlib.Path(__file__).resolve().parents[1]/"results/validation/phase2_derivation_checks.txt"
p.parent.mkdir(parents=True,exist_ok=True); p.write_text(txt+"\n")
assert all(l.startswith(("PASS","INFO")) for l in out), "a derivation check failed"
