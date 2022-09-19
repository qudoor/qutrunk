include "qelib1.inc";

// gate r(theta, phi) a
// {
//     U(theta,phi+(3*pi)/2,-phi+pi/2) a;
// }

// todo to be verify
// gate sqrtswap a,b
// {
//     cx b,a;
//     ry(pi/2) b;
//     rz(pi/16) b;
//     cx a,b;
//     rz(pi/8) a;
//     rz(-pi/8) b;
//     cx a,b;
//     rz(pi/16) b;
//     ry(-pi/2) b;
//     cx b,a;
// }

// gate sqrtx a
// {
//     sx a;
// }

// todo to be verify
// gate ccz a,b,c
// {
//     cx b,c;
//     tdg c;
//     cx a,c;
//     t c;
//     cx b,c;
//     tdg c;
//     cx a,c;
//     t b;
//     t c;
//     cx a,b;
//     t a;
//     tdg b;
//     cx a,b;
// }

opaque r(theta, phi) a;
opaque sqrtswap a,b;
opaque sqrtx a;
opaque ccz a,b,c;
opaque ryy(theta) a,b;
opaque x1 a;
opaque y1 a;
opaque z1 a;
opaque iswap(theta) a,b;
opaque cr(theta) a,b;
