OPENQASM 2.0;
// some gate only specified by qutruk, not OpenQASM
// qulib1.inc add these gate as opaque gate
// qulib1.inc also include "qelib1.inc"
include "qulib1.inc";

qreg q[5];
creg c[5];

// custom gate
gate myH a
{
    h a;
}

// nested custom gate
gate myHH b
{
    myH b;
}

myH q[0];
myHH q[1];

h q[0];
cx q[0],q[1];
x q[0];
ccx q[0],q[1],q[2];
p(pi/2) q[2];
r(pi/2,pi/2) q[0];
rx(pi/2) q[1];
ry(pi/2) q[1];
rz(pi/2) q[1];
s q[0];
sdg q[0];
t q[0];
tdg q[0];
x q[2];
y q[2];
z q[2];
x1 q[0];
y1 q[0];
z1 q[0];
swap q[0],q[1];
iswap(pi/2) q[0],q[1];
sqrtswap q[0],q[1];
sqrtx q[0];
cx q[0],q[1];
cy q[0],q[1];
cz q[0],q[1];
cp(pi/2) q[0],q[1];
cr(pi/2) q[0],q[1];
crx(pi/2) q[0],q[1];
cry(pi/2) q[0],q[1];
crz(pi/2) q[0],q[1];
ccx q[0],q[1],q[2];
ccz q[0],q[1],q[2];
rxx(pi/2) q[0],q[1];
ryy(pi/2) q[0],q[1];
rzz(pi/2) q[0],q[1];
u1(pi/2) q[0];
u2(pi/2,pi/2) q[0];
u3(pi/2,pi/2,pi/2) q[0];
cu(pi/2,pi/2,pi/2,pi/2) q[0],q[1];
cu1(pi/2) q[1],q[2];
cu3(pi/2,pi/2,pi/2) q[0],q[1];
id q[0];
ch q[0],q[1];
cswap q[0],q[1],q[2];
csx q[0],q[1];
sxdg q[0];
barrier q[0],q[1],q[2];

// gate from OpenQASM
c4x q[0],q[1],q[2],q[3],q[4];

measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];