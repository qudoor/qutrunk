#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <time.h>
#include <fstream>
#include <memory>
#include <algorithm>
#include <sstream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "QuEST/include/QuEST.h"

#ifndef M_PI 
#define M_PI 3.1415926535897932384626433832795028841971
#endif

using namespace std;
namespace py = pybind11;

using Qubits = vector<int>;
using Qurota = vector<double>;

typedef struct _Amplitude{
    vector<double> reals;
    vector<double> imags;
    int startind;
    int numamps;
} Amplitude;

typedef struct _CmdEx {
    Amplitude amp;
} CmdEx;

typedef struct _Cmd {
    string gate{""};
    vector<int>  targets;
    vector<int>  controls;
    vector<double>  rotation;
    string desc{""};
    bool inverse{false};
    CmdEx cmdex;
} Cmd;

using Circuit = vector<Cmd>;

typedef struct _MeasureResult {
    // 量子比特id
    int id{0};
    // 测量结果
    int value{0};
} MeasureResult;

using MeasureSet = vector<MeasureResult>;

typedef struct _Outcome {
    // 比特位字符串 "000", "001"...
    string bitstr{""};
    // 比特位字符串出现的次数
    int count{0};
} Outcome;

using OutcomeSet = vector<Outcome>;

typedef struct _Result {
    MeasureSet measureSet;
    OutcomeSet outcomeSet;
} Result;

//泡利算子操作类型oper_type,PAULI_I = 0,  PAULI_X = 1,  PAULI_Y = 2,  PAULI_Z = 3
struct PauliProdInfo {
    //泡利算子操作类型
    int oper_type{0};

    //目标比特位
    int target{0};
};
using PauliProdList = vector<PauliProdInfo>;

using GateFunc = void (*)(const Cmd& cmd);

QuESTEnv g_env;
Qureg    g_qureg;
bool     g_bshowquantumgate = false;
Circuit  g_circuit;
Result   g_result;
int      g_qubits = 0;

extern "C" void invalidQuESTInputError(const char* errMsg, const char* errFunc) {
 
    string err = "in function " + string(errFunc) + ": " + string(errMsg);
    throw std::invalid_argument(err);
} 

void qh(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }
    
    if (cmd.inverse) {
       //逆操作和正常操作一样
    }

    hadamard(g_qureg, cmd.targets[0]);
}

void qch(const Cmd& cmd)
{
    if (cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    } 

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    double factor = 1 / sqrt(2);
    ComplexMatrix2 m = {
            {
                {1 * factor, 1 * factor}, 
                {1 * factor, -1 * factor}
            },
            {
                {0, 0}, 
                {0, 0}
            }
    };

    controlledUnitary(g_qureg, cmd.controls[0], cmd.targets[0], m);
}

void qp(const Cmd& cmd)
{
    // matrix([[cmath.exp(1j * self.angle), 0], [0, cmath.exp(1j * self.angle)]]
    // if (targets.size() != 1) {
    //     return;
    // }

    // double angle = rotation[0];
    // ComplexMatrix2 m = {
    //     {{cos(angle), 0}, {0, cos(angle)}},
    //     {{sin(angle), 0}, {0, sin(angle)}}};

    // applyMatrix2(g_qureg, targets[0], m);

    if (cmd.targets.size() != 1 || cmd.rotation.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    int targetQubit = cmd.targets[0];
    phaseShift(g_qureg, targetQubit, rotation); 
}

void qcp(const Cmd& cmd)
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    controlledPhaseShift(g_qureg, cmd.controls[0], cmd.targets[0], rotation);
}

void qr(const Cmd& cmd)
{
    if (cmd.targets.size() != 1 || cmd.rotation.size() != 2) {
        return;
    }

    // phaseShift(g_qureg, targets[0], rotation[0]);
    // matrix
    // theta, phi = float(self.alpha), float(self.beta)
    // cos = math.cos(theta / 2)
    // sin = math.sin(theta / 2)
    // exp_m = numpy.exp(-1j * phi) = cos(phi)-sin(phi)
    // exp_p = numpy.exp(1j * phi) = cos(phi)+sin(phi)
    // numpy.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]])

    auto theta = cmd.rotation[0];
    if (cmd.inverse) {
        theta = -theta;
    }
    auto phi = cmd.rotation[1];

    ComplexMatrix2 m = {
        {
            {cos(theta/2), sin(-phi) * sin(theta/2)}, 
            {sin(phi) * sin(theta/2), cos(theta/2)}
        },
        {
            {0, -1 * cos(-phi) * sin(theta/2)}, 
            {-1 * cos(phi) * sin(theta/2), 0}
        }
    };

    applyMatrix2(g_qureg, cmd.targets[0], m);
}

void qrx(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    rotateX(g_qureg, cmd.targets[0], rotation);
}

void qry(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    rotateY(g_qureg, cmd.targets[0], rotation);
}

void qrz(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    rotateZ(g_qureg, cmd.targets[0], rotation);
}

void qrxx(const Cmd& cmd)
{
    // projectq matrix
    // matrix(
    //         [
    //             [cmath.cos(0.5 * self.angle), 0, 0, -1j * cmath.sin(0.5 * self.angle)],
    //             [0, cmath.cos(0.5 * self.angle), -1j * cmath.sin(0.5 * self.angle), 0],
    //             [0, -1j * cmath.sin(0.5 * self.angle), cmath.cos(0.5 * self.angle), 0],
    //             [-1j * cmath.sin(0.5 * self.angle), 0, 0, cmath.cos(0.5 * self.angle)],
    //         ]
    //     )
    if (cmd.targets.size() != 2) {
        return;
    }

    double angle = cmd.rotation[0];
    if (cmd.inverse) {
        angle = -angle;
    }

    ComplexMatrix4 m = {
            {{cos(0.5 * angle), 0, 0, 0},
            {0, cos(0.5 * angle), 0, 0},
            {0, 0, cos(0.5 * angle), 0},
            {0, 0, 0, cos(0.5 * angle)}
        },
            {
            {0, 0, 0, -1 * sin(0.5 * angle)},
            {0, 0, -1 * sin(0.5 * angle), 0},
            {0, -1 * sin(0.5 * angle), 0, 0},
            {-1 * sin(0.5 * angle), 0, 0, 0}
        }
    };

    applyMatrix4(g_qureg, cmd.targets[0], cmd.targets[1], m);
}

void qryy(const Cmd& cmd)
{
    // projectq matrix
    // matrix(
    //     [
    //         [cmath.cos(0.5 * self.angle), 0, 0, 1j * cmath.sin(0.5 * self.angle)],
    //         [0, cmath.cos(0.5 * self.angle), -1j * cmath.sin(0.5 * self.angle), 0],
    //         [0, -1j * cmath.sin(0.5 * self.angle), cmath.cos(0.5 * self.angle), 0],
    //         [1j * cmath.sin(0.5 * self.angle), 0, 0, cmath.cos(0.5 * self.angle)],
    //     ]
    // )

    if (cmd.targets.size() != 2) {
        return;
    }
    double angle = cmd.rotation[0];
    if (cmd.inverse) {
        angle = -angle;
    }
    ComplexMatrix4 m = {
            {{cos(0.5 * angle), 0, 0, 0},
            {0, cos(0.5 * angle), 0, 0},
            {0, 0, cos(0.5 * angle), 0},
            {0, 0, 0, cos(0.5 * angle)}
        },
            {
            {0, 0, 0, 1 * sin(0.5 * angle)},
            {0, 0, -1 * sin(0.5 * angle), 0},
            {0, -1 * sin(0.5 * angle), 0, 0},
            {1 * sin(0.5 * angle), 0, 0, 0}
        }
    };

    applyMatrix4(g_qureg, cmd.targets[0], cmd.targets[1], m);
}

void qrzz(const Cmd& cmd)
{
    // projectq matrix
    // matrix(
    //     [
    //         [cmath.cos(0.5 * self.angle), 0, 0, 1j * cmath.sin(0.5 * self.angle)],
    //         [0, cmath.cos(0.5 * self.angle), -1j * cmath.sin(0.5 * self.angle), 0],
    //         [0, -1j * cmath.sin(0.5 * self.angle), cmath.cos(0.5 * self.angle), 0],
    //         [1j * cmath.sin(0.5 * self.angle), 0, 0, cmath.cos(0.5 * self.angle)],
    //     ]
    // )

    if (cmd.targets.size() != 2) {
        return;
    }

    double angle = cmd.rotation[0];
    if (cmd.inverse) {
        angle = -angle;
    }
    ComplexMatrix4 m = {
        {
            {cos(0.5 * angle), 0, 0, 0},
            {0, cos(0.5 * angle), 0, 0},
            {0, 0, cos(0.5 * angle), 0},
            {0, 0, 0, cos(0.5 * angle)}
        },
        {
            {-1 * sin(0.5 * angle), 0, 0, 0},
            {0, sin(0.5 * angle), 0, 0},
            {0, 0, sin(0.5 * angle), 0},
            {0, 0, 0, -1 * sin(0.5 * angle)}
        }
    };

    applyMatrix4(g_qureg, cmd.targets[0], cmd.targets[1], m);
}

void qx(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    pauliX(g_qureg, cmd.targets[0]);
}

void qy(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    pauliY(g_qureg, cmd.targets[0]);
}

void qz(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    pauliZ(g_qureg, cmd.targets[0]);
}

void qsdg(const Cmd& cmd);
void qs(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        Cmd temp(cmd);
        temp.inverse = false;
        qsdg(temp);
    } else {
        sGate(g_qureg, cmd.targets[0]);
    }
}

void qtdg(const Cmd& cmd);
void qt(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        Cmd temp(cmd);
        temp.inverse = false;
        qtdg(temp);
    } else {
        tGate(g_qureg, cmd.targets[0]);
    }
}

void qsdg(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        Cmd temp(cmd);
        temp.inverse = false;
        qs(temp);
    } else {
        // sgate matrix([[1, 0], [0, 1j]])
        // 共轭转置：matrix([[1, 0], [0, -1j]])
        ComplexMatrix2 m = {
            {
                {1, 0},
                {0, 0}
            },
            {
                {0, 0},
                {0, -1}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m); 
    }
}

void qtdg(const Cmd& cmd)
{
     if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        Cmd temp(cmd);
        temp.inverse = false;
        qt(temp);
    } else {
        // matrix([[1, 0], [0, cmath.exp(1j * cmath.pi / 4)]])
        // 共轭转置：matrix([[1, 0], [0, cmath.exp(-1j * cmath.pi / 4)]])
        ComplexMatrix2 m = {
            {
                {1, 0},
                {0, 1/sqrt(2)}
            },
            {
                {0, 0},
                {0, -1/sqrt(2)}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m); 
    }
}

void qsqrtx(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        ComplexMatrix2 m = {
            {
                {0.5, 0.5},
                {0.5, 0.5}
            },
            {
                {-0.5, 0.5},
                {0.5, -0.5}
            }
        };
        applyMatrix2(g_qureg, cmd.targets[0], m);
    } else {
        // projectq matrix
        // 0.5 * np.matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        ComplexMatrix2 m = {
            {
                {0.5, 0.5},
                {0.5, 0.5}
            },
            {
                {0.5, -0.5},
                {-0.5, 0.5}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m);
    }
}

void qcsqrtx(const Cmd& cmd)
{
    if (cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    } 

    if (cmd.inverse) {
        ComplexMatrix2 m = {
            {
                {0.5, 0.5},
                {0.5, 0.5}
            },
            {
                {-0.5, 0.5},
                {0.5, -0.5}
            }
        };
        controlledUnitary(g_qureg, cmd.controls[0], cmd.targets[0], m);
    } else {
        // projectq matrix
        // 0.5 * np.matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        ComplexMatrix2 m = {
            {
                {0.5, 0.5},
                {0.5, 0.5}
            },
            {
                {0.5, -0.5},
                {-0.5, 0.5}
            }
        };

        controlledUnitary(g_qureg, cmd.controls[0], cmd.targets[0], m);
    }
}

void qsqrtxdg(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) 
    {
        // projectq matrix
        // 0.5 * np.matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
        ComplexMatrix2 m = {
            {
                {0.5, 0.5},
                {0.5, 0.5}
            },
            {
                {0.5, -0.5},
                {-0.5, 0.5}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m);
    } 
    else 
    {
        ComplexMatrix2 m = {
            {
                {0.5, 0.5},
                {0.5, 0.5}
            },
            {
                {-0.5, 0.5},
                {0.5, -0.5}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m);
    }
}

void qsqrtswap(const Cmd& cmd)
{
    if (cmd.targets.size() != 2) {
        return;
    }

    if (cmd.inverse) {
        // matrix(
        //     [
        //         [1, 0, 0, 0],
        //         [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
        //         [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
        //         [0, 0, 0, 1],
        //     ]
        // )
        ComplexMatrix4 m = {
            {{1, 0, 0, 0},
                {0, 0.5, 0.5, 0},
                {0, 0.5, 0.5, 0},
                {0, 0, 0, 1}
            },
            {
                {0, 0, 0, 0},
                {0, -0.5, 0.5, 0},
                {0, 0.5, -0.5, 0},
                {0, 0, 0, 0}
            }
        };

        applyMatrix4(g_qureg, cmd.targets[0], cmd.targets[1], m);
    } else {
        // matrix(
        //     [
        //         [1, 0, 0, 0],
        //         [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
        //         [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
        //         [0, 0, 0, 1],
        //     ]
        // )
        ComplexMatrix4 m = {
            {{1, 0, 0, 0},
                {0, 0.5, 0.5, 0},
                {0, 0.5, 0.5, 0},
                {0, 0, 0, 1}
            },
            {
                {0, 0, 0, 0},
                {0, 0.5, -0.5, 0},
                {0, -0.5, 0.5, 0},
                {0, 0, 0, 0}
            }
        };

        applyMatrix4(g_qureg, cmd.targets[0], cmd.targets[1], m);
    }
}

void qswap(const Cmd& cmd)
{
    if (cmd.targets.size() != 2) {
        return;
    }

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    swapGate(g_qureg, cmd.targets[0], cmd.targets[1]);
}

void qcswap(const Cmd& cmd)
{
    if (cmd.targets.size() != 2 || cmd.controls.size() != 1) {
        return;
    } 

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    ComplexMatrix4 m = {
            {
                {1, 0, 0, 0}, 
                {0, 0, 1, 0},
                {0, 1, 0, 0},
                {0, 0, 0, 1}
            },
            {
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}
            }
    };

    controlledTwoQubitUnitary(g_qureg, cmd.controls[0], cmd.targets[0], cmd.targets[1], m);
}

void qcnot(const Cmd& cmd)
{
    if (cmd.targets.size() == 1 && cmd.controls.size() == 1) {
        if (cmd.inverse) {
            //逆操作和正常操作一样
        }
        controlledNot(g_qureg, cmd.controls[0], cmd.targets[0]);
    } else {
        auto ctrls = make_unique<int[]>(cmd.controls.size());
        copy(cmd.controls.begin(), cmd.controls.end(), ctrls.get());
        auto targs = make_unique<int[]>(cmd.targets.size());
        copy(cmd.targets.begin(), cmd.targets.end(), targs.get());
        if (cmd.inverse) {
            //逆操作和正常操作一样
        }
        multiControlledMultiQubitNot(g_qureg, 
            ctrls.get(), 
            cmd.controls.size(), 
            targs.get(), 
            cmd.targets.size()
        ); 
    }
}

void qcy(const Cmd& cmd)
{
    if (cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    } 

    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    controlledPauliY(g_qureg, cmd.controls[0], cmd.targets[0]);
}

void qcz(const Cmd& cmd)
{
   // cmd.controls.insert(cmd.controls.cend(), cmd.targets.cbegin(), cmd.targets.cend());
    vector<int32_t> tempCtls;
    tempCtls.assign(cmd.controls.begin(), cmd.controls.end());
    tempCtls.insert(tempCtls.end(), cmd.targets.begin(), cmd.targets.end());
    auto ctrls = make_unique<int[]>(tempCtls.size());
    copy(tempCtls.begin(), tempCtls.end(), ctrls.get());


    if (cmd.inverse) {
        //逆操作和正常操作一样
    }

    multiControlledPhaseFlip(g_qureg, 
        ctrls.get(), 
        (int)tempCtls.size()
    );
}

void qu3(const Cmd& cmd)
{
   if (cmd.rotation.size() != 3 || cmd.targets.size() != 1) {
        return;
    }

    auto theta = cmd.rotation[0];
    auto phi = cmd.rotation[1];
    auto lam = cmd.rotation[2];
    if (cmd.inverse) {
        auto theta1 = -theta;
        auto phi1 = -lam;
        auto lam1 = -phi;
        theta = theta1;
        phi = phi1;
        lam = lam1;
    }
    
    ComplexMatrix2 m = {
        {
            {cos(theta / 2), -1 * cos(lam) * sin(theta / 2) },
            {cos(phi) * sin(theta / 2) , cos(phi + lam) * cos(theta / 2)}
        },
        {
            {0, -1 * sin(lam) * sin(theta / 2)},
            {sin(phi) * sin(theta / 2), sin(phi + lam) * cos(theta / 2)}
        }
    };

    unitary(g_qureg, cmd.targets[0], m);
}

void qu2(const Cmd& cmd)
{
   if (cmd.rotation.size() != 2 || cmd.targets.size() != 1) {
        return;
    }

    auto phi = cmd.rotation[0];
    auto lam = cmd.rotation[1];

    if (cmd.inverse) {
        auto phi1 = -lam - M_PI;
        auto lam1 = -phi + M_PI;
        phi = phi1;
        lam = lam1;
    }

    double factor = 1 / sqrt(2);
    ComplexMatrix2 m = {
        {
            {1 * factor, -factor * cos(lam)},
            {factor * cos(phi), factor * cos(phi + lam)}
        },
        {
            {0, -factor * sin(lam)},
            {factor * sin(phi), factor * sin(phi + lam)}
        }
    };

    unitary(g_qureg, cmd.targets[0], m);
}

void qu1(const Cmd& cmd)
{
   if (cmd.rotation.size() != 1 || cmd.targets.size() != 1) {
        return;
    }

    auto alpha = cmd.rotation[0];
    if (cmd.inverse) {
        alpha = -alpha;
    }

    ComplexMatrix2 m = {
        {
            {1, 0},
            {0, cos(alpha)}
        },
        {
            {0, 0},
            {0, sin(alpha)}
        }
    };
    unitary(g_qureg, cmd.targets[0], m);
}


void qcrx(const Cmd& cmd) 
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    controlledRotateX(g_qureg, cmd.controls[0], cmd.targets[0], rotation);
}

void qcry(const Cmd& cmd) 
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    controlledRotateY(g_qureg, cmd.controls[0], cmd.targets[0], rotation);
}

void qcrz(const Cmd& cmd) 
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    double rotation = cmd.rotation[0];
    if (cmd.inverse) {
        rotation = -rotation;
    }

    controlledRotateZ(g_qureg, cmd.controls[0], cmd.targets[0], rotation);
}

int qmeasure(const int qubit)
{
    return measure(g_qureg, qubit);
}

void qx1(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        double factor = sqrt(2);
        ComplexMatrix2 m = {
            {
                {0.5 * factor, 0}, 
                {0, 0.5 * factor}
            },
            {
                {0, 0.5 * factor}, 
                {0.5 * factor, 0}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m); 
    } else {
        double factor = 1 / sqrt(2);
        ComplexMatrix2 m = {
            {
                {1 * factor, 0}, 
                {0, 1 * factor}
            },
            {
                {0, -1 * factor}, 
                {-1 * factor, 0}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m); 
    }
}

void qy1(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    if (cmd.inverse) {
        double factor = sqrt(2);
        ComplexMatrix2 m = {
            {
                {0.5 * factor, 0.5 * factor}, 
                {-0.5 * factor, 0.5 * factor}
            },
            {
                {0, 0}, 
                {0, 0}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m); 
    } else {
        double factor = 1 / sqrt(2);
        ComplexMatrix2 m = {
            {
                {1 * factor, -1 * factor}, 
                {1 * factor, 1 * factor}
            },
            {
                {0, 0}, 
                {0, 0}
            }
        };

        applyMatrix2(g_qureg, cmd.targets[0], m); 
    }
}

void qz1(const Cmd& cmd)
{
    if (cmd.targets.size() != 1) {
        return;
    }

    double rotation = M_PI/4.0;
    if (cmd.inverse) {
        rotation = -rotation;
    } 

    ComplexMatrix2 m = {
        {
            {cos(-rotation), 0}, 
            {0, cos(rotation)}
        },
        {
            {sin(-rotation), 0}, 
            {0, sin(rotation)}
        }
    };

    applyMatrix2(g_qureg, cmd.targets[0], m); 
}

void qcu1(const Cmd& cmd)
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    auto alpha = cmd.rotation[0];

    if (cmd.inverse) {
        alpha = -alpha;
    }

    ComplexMatrix4 m = {
        {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, cos(alpha)}
        },
        {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, sin(alpha)}
        }
    };
    applyMatrix4(g_qureg, cmd.controls[0], cmd.targets[0], m);
}

void qcu3(const Cmd& cmd)
{
    if (cmd.rotation.size() != 3 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    auto theta = cmd.rotation[0];
    auto phi = cmd.rotation[1];
    auto lam = cmd.rotation[2];
    if (cmd.inverse) {
        auto theta1 = -theta;
        auto phi1 = -lam;
        auto lam1 = -phi;
        theta = theta1;
        phi = phi1;
        lam = lam1;
    }

    ComplexMatrix4 m = {
        {
            {1, 0, 0, 0},
            {0, cos(theta/2), 0, -cos(lam)*sin(theta/2)},
            {0, 0, 1, 0},
            {0, cos(phi)*sin(theta/2), 0, cos(phi + lam)*cos(theta/2)}
        },
        {
            {0, 0, 0, 0},
            {0, 0, 0, -sin(lam)*sin(theta/2)},
            {0, 0, 0, 0},
            {0, sin(phi)*sin(theta/2), 0, sin(phi + lam)*cos(theta/2)}
        }
    };
    applyMatrix4(g_qureg, cmd.controls[0], cmd.targets[0], m);
}

void qu(const Cmd& cmd)
{
    qu3(cmd);
}

void qcu(const Cmd& cmd)
{
    if (cmd.rotation.size() != 4 || cmd.targets.size() != 1 || cmd.controls.size() != 1) {
        return;
    }

    auto theta = cmd.rotation[0];
    auto phi = cmd.rotation[1];
    auto lam = cmd.rotation[2];
    auto gamma = cmd.rotation[3];
    if (cmd.inverse) {
        auto theta1 = -theta;
        auto phi1 = -lam;
        auto lam1 = -phi;
        auto gamma1 = -gamma;
        theta = theta1;
        phi = phi1;
        lam = lam1;
        gamma = gamma1;
    }

    ComplexMatrix4 m = {
        {
            {1, 0, 0, 0},
            {0, cos(gamma)*cos(theta/2), 0, -cos(gamma + lam)*sin(theta/2)},
            {0, 0, 1, 0},
            {0, cos(gamma + phi)*sin(theta/2), 0, cos(gamma + phi + lam)*cos(theta/2)}
        },
        {
            {0, 0, 0, 0},
            {0, sin(gamma)*cos(theta/2), 0, -sin(gamma + lam)*sin(theta/2)},
            {0, 0, 0, 0},
            {0, sin(gamma + phi)*sin(theta/2), 0, sin(gamma + phi + lam)*cos(theta/2)}
        }
    };
    applyMatrix4(g_qureg, cmd.controls[0], cmd.targets[0], m);
}

void qcr(const Cmd& cmd)
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 1 || cmd.controls.size() !=1) {
        return;
    }

    auto theta = cmd.rotation[0];
    if (cmd.inverse) {
        theta = -theta;
    }

    ComplexMatrix4 m = {
        {
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, cos(theta)}
        },
        {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, sin(theta)}
        }
    };
    applyMatrix4(g_qureg, cmd.controls[0], cmd.targets[0], m);
}

void qiswap(const Cmd& cmd)
{
    if (cmd.rotation.size() != 1 || cmd.targets.size() != 2) {
        return;
    }

    auto theta = cmd.rotation[0];
    if (cmd.inverse) {
        theta = -theta;
    }

    ComplexMatrix4 m = {
        {
            {1, 0, 0, 0},
            {0, cos(theta), 0, 0},
            {0, 0, cos(theta), 0},
            {0, 0, 0, 1}
        },
        {
            {0, 0, 0, 0},
            {0, 0, -sin(theta), 0},
            {0, -sin(theta), 0, 0},
            {0, 0, 0, 0}
        }
    };
    applyMatrix4(g_qureg, cmd.targets[0], cmd.targets[1], m);
}

void qid(const Cmd& cmd)
{
    // do nothing
}

void qamp(const Cmd& cmd)
{
    setAmps(g_qureg, cmd.cmdex.amp.startind, (double*)cmd.cmdex.amp.reals.data(), (double*)cmd.cmdex.amp.imags.data(), cmd.cmdex.amp.numamps);
}

map<string, GateFunc> g_gateMap = {
    make_pair<string, GateFunc>("H", &qh),
    make_pair<string, GateFunc>("P", &qp),
    make_pair<string, GateFunc>("CP", &qcp),
    make_pair<string, GateFunc>("R", &qr),
    make_pair<string, GateFunc>("Rx", &qrx), 
    make_pair<string, GateFunc>("Ry", &qry),
    make_pair<string, GateFunc>("Rz", &qrz),
    make_pair<string, GateFunc>("Rxx", &qrxx),
    make_pair<string, GateFunc>("Ryy", &qryy),
    make_pair<string, GateFunc>("Rzz", &qrzz),
    make_pair<string, GateFunc>("NOT", &qx),
    make_pair<string, GateFunc>("X", &qx),
    make_pair<string, GateFunc>("Y", &qy),
    make_pair<string, GateFunc>("Z", &qz),
    make_pair<string, GateFunc>("S", &qs),
    make_pair<string, GateFunc>("T", &qt),
    make_pair<string, GateFunc>("Sdg", &qsdg),
    make_pair<string, GateFunc>("Tdg", &qtdg),
    make_pair<string, GateFunc>("SqrtX", &qsqrtx),
    make_pair<string, GateFunc>("SqrtSwap", &qsqrtswap),
    make_pair<string, GateFunc>("Swap", &qswap),
    make_pair<string, GateFunc>("CNOT", &qcnot),
    make_pair<string, GateFunc>("MCX", &qcnot),
    make_pair<string, GateFunc>("CY", &qcy),
    make_pair<string, GateFunc>("MCZ", &qcz),
    make_pair<string, GateFunc>("U3", &qu3),
    make_pair<string, GateFunc>("U2", &qu2),
    make_pair<string, GateFunc>("U1", &qu1),
    make_pair<string, GateFunc>("CRx", &qcrx),
    make_pair<string, GateFunc>("CRy", &qcry),
    make_pair<string, GateFunc>("CRz", &qcrz),
    make_pair<string, GateFunc>("X1", &qx1),
    make_pair<string, GateFunc>("Y1", &qy1),
    make_pair<string, GateFunc>("Z1", &qz1),
    make_pair<string, GateFunc>("CU1", &qcu1),
    make_pair<string, GateFunc>("CU3", &qcu3),
    make_pair<string, GateFunc>("U", &qu),
    make_pair<string, GateFunc>("CU", &qcu),
    make_pair<string, GateFunc>("CR", &qcr),
    make_pair<string, GateFunc>("iSwap", &qiswap),
    make_pair<string, GateFunc>("Id", &qid), 
    make_pair<string, GateFunc>("CH", &qch), 
    make_pair<string, GateFunc>("SqrtXdg", &qsqrtxdg), 
    make_pair<string, GateFunc>("CSqrtX", &qcsqrtx), 
    make_pair<string, GateFunc>("CSwap", &qcswap),
    make_pair<string, GateFunc>("AMP", &qamp)
};

void init(int qubits, bool showquantumgate) {
    g_env = createQuESTEnv();
    g_qureg = createQureg(qubits, g_env);
    g_bshowquantumgate = showquantumgate;
    g_circuit.clear();
    g_result.measureSet.clear();
    g_result.outcomeSet.clear();

    g_qubits = qubits;

    if (g_bshowquantumgate) {
        startRecordingQASM(g_qureg);
        cout << "################# qutrunk Cmd Start! #################" << endl;
        cout << "qreg q[" << qubits << "]" << endl;
        cout << "creg c[" << qubits << "]" << endl;
    }

    initZeroState(g_qureg);
}

void release() {
    if (g_bshowquantumgate) {
        cout << "################# qutrunk Cmd End! ###################" << endl;
        cout << "***************** QuEST Cmd Start! *****************" << endl;
        printRecordedQASM(g_qureg);
        cout << "***************** QuEST Cmd End! *****************" << endl;
        clearRecordedQASM(g_qureg);
        stopRecordingQASM(g_qureg);
    }
    destroyQureg(g_qureg, g_env);
    destroyQuESTEnv(g_env);
}

// todo: 直接传cmd
int execCmd(const Cmd& cmd)
{
    if (cmd.gate == "Measure") {
        int res = qmeasure(cmd.targets[0]);
        MeasureResult mr;
        mr.id = cmd.targets[0];
        mr.value = res;
        g_result.measureSet.push_back(mr);
    }

    const auto it = g_gateMap.find(cmd.gate);
    if (it == g_gateMap.end()) {
        return -1;
    }

    if (g_bshowquantumgate) {
        cout << cmd.desc << endl;
    }

    try
    {
        (*(it->second))(cmd);
    }
    catch(invalid_argument& ex)
    {
        std::cout << "(" << cmd.gate << ")" << " Invalid argument exception " << ex.what();
        return -1;
    }
    catch(exception& ex)
    {
        std::cout << "(" << cmd.gate << ")" << " Unknown exception " << ex.what();
        return -1;
    }
    
    return 0;
}

void pack_measrue_result() {
    sort(g_result.measureSet.begin(), g_result.measureSet.end(), [](const MeasureResult& a, const MeasureResult& b) {return a.id < b.id;});
    string bitstr = "";
    for (auto &m : g_result.measureSet) {
        int val = m.value;
        bitstr.append(to_string(val));
    }

    int idx = -1;
    for (int i = 0; i < g_result.outcomeSet.size(); ++i) {
        if (g_result.outcomeSet[i].bitstr == bitstr) {
            idx = i;
            break; 
        }
    }
    if (idx >= 0) {
        g_result.outcomeSet[idx].count += 1;
    } else {
        Outcome out;
        out.bitstr = bitstr;
        out.count = 1;
        g_result.outcomeSet.push_back(out);
    }
}

void send_circuit(const Circuit& circuit, const bool final) {
    for (const auto & cmd : circuit) {
        // 缓存指令
        g_circuit.push_back(cmd);
        // 执行指令
        execCmd(cmd);
    }

    if (final) {
        pack_measrue_result();
    }
}

Result run(int shots) {
    // 如果只运行一次，那么在线路发送完成后已经自动运行一次了
    int run_times = shots - 1;
    while (run_times > 0) {
        g_result.measureSet.clear();
        
        // 每次运行需要重置状态
        initZeroState(g_qureg);

        for (const auto & cmd : g_circuit) {
            // 执行指令
            execCmd(cmd);
        }
        pack_measrue_result();
        run_times -= 1;
    }

    release();

    sort(g_result.outcomeSet.begin(), g_result.outcomeSet.end(), [](const Outcome& a, const Outcome& b) {return a.bitstr.compare(b.bitstr) < 0;});

    return g_result;
}

double getProbOfAmp(const int64_t index) {
    return getProbAmp(g_qureg, index);
}

double getProbOfOutcome(const int32_t qubit, const int32_t outcome) {
    return calcProbOfOutcome(g_qureg, qubit, outcome);
}

std::vector<double> getProbOfAllOutcome(const std::vector<int32_t> & qubits) {
    std::vector<double> qalloutcome;

    int numQubits = (int)qubits.size();
    if (numQubits == 0) {
        return qalloutcome;
    }

    int numOutcomes = 1LL << numQubits;

    auto probs = make_unique<qreal[]>(numOutcomes);
    auto bits = make_unique<int[]>(numOutcomes);

    copy(qubits.begin(), qubits.end(), bits.get());
    calcProbOfAllOutcomes(probs.get(), g_qureg, bits.get(), numQubits);

    for (auto i = 0; i < numOutcomes; ++i) {
        qalloutcome.push_back(probs[i]);
    }

    return qalloutcome;
}

std::vector<std::string> getAllState() {
    std::vector<std::string> qallstate;

    std::stringstream ss;
    ss.precision(12);
    for(long long index = 0; index < g_qureg.numAmpsPerChunk; index++)
    {
        ss.str("");
        qreal real = g_qureg.stateVec.real[index];
        qreal imag = g_qureg.stateVec.imag[index];
        if (real > -1e-15 && real < 1e-15)
        {
            real = 0;
        }
        if (imag > -1e-15 && imag < 1e-15)
        {
            imag = 0;
        }
        ss << real << ", " << imag;
        qallstate.push_back(ss.str());
    }

    return qallstate;
}

void apply_QFT(const std::vector<int32_t> & qubits) {
    if (qubits.size() == 0) {
        return ;
    }

    auto bits = std::make_unique<int32_t []>(qubits.size());

    copy(qubits.begin(), qubits.end(), bits.get());
    
    applyQFT(g_qureg, bits.get(), (int)qubits.size());
}

void apply_Full_QFT() {
    applyFullQFT(g_qureg);
}

double getExpecPauliProd(const PauliProdList& pauli_prod) {
    size_t prodsize = pauli_prod.size();
    if (0 == prodsize) {
        cout << "getExpecPauliProd Invaild parameters prodsize is empty" << endl;
        return 0;
    }

    Qureg workqureg;
    workqureg = createQureg(g_qubits, g_env);

    std::unique_ptr<int []> targets(new int[prodsize]());
    std::unique_ptr<pauliOpType []> paulitype(new pauliOpType[prodsize]());
    for (size_t i = 0; i < prodsize; ++i) {
        targets[i] = pauli_prod[i].target;
        paulitype[i] = (pauliOpType)pauli_prod[i].oper_type;
    }
    double expectvalue = calcExpecPauliProd(g_qureg, targets.get(), paulitype.get(), prodsize, workqureg);

    destroyQureg(workqureg, g_env);
    return expectvalue;

}

double getExpecPauliSum(const std::vector<int32_t>& pauli_type, const std::vector<double>& coeff_type) {
    size_t typesize = pauli_type.size();
    size_t coeffize = coeff_type.size();
    if (coeffize * g_qubits != typesize) {
        cout << "getExpecPauliSum Invaild parameters(typesize:" << typesize << ",coeffize:" << coeffize << ",qubitsnum:" << g_qubits << ")" << endl;
        return 0;
    }

    Qureg workqureg;
    workqureg = createQureg(g_qubits, g_env);

    std::unique_ptr<pauliOpType []> paulitype(new pauliOpType[typesize]());
    std::unique_ptr<qreal []> coeffs(new qreal[coeffize]());
    for (size_t i = 0; i < typesize; ++i) {
        paulitype[i] = (pauliOpType)pauli_type[i];
    }
    for (size_t i = 0; i < coeffize; ++i) {
        coeffs[i] = coeff_type[i];
    }

    double expectvalue = calcExpecPauliSum(g_qureg, paulitype.get(), coeffs.get(), coeffize, workqureg);

    destroyQureg(workqureg, g_env);
    return expectvalue;
}

PYBIND11_MODULE(simulator, m) {
    m.doc() = "simulator plugin";

    m.def("init", &init, "init sim envirement", py::arg("qubits"), py::arg("showquantumgate"));
    m.def("getProbOfAmp", &getProbOfAmp, "getProbOfAmp", py::arg("index"));
    m.def("getProbOfOutcome", &getProbOfOutcome, "getProbOfOutcome", py::arg("qubit"), py::arg("outcome"));
    m.def("getProbOfAllOutcome", &getProbOfAllOutcome, "getProbOfAllOutcome", py::arg("qubits"));
    m.def("getAllState", &getAllState, "getAllState");
    m.def("apply_QFT", &apply_QFT, "apply_QFT", py::arg("qubits"));
    m.def("apply_Full_QFT", &apply_Full_QFT, "apply_Full_QFT");

    m.def("send_circuit", &send_circuit, "send_circuit", py::arg("circuit"), py::arg("final"));
    m.def("run", &run, "run", py::arg("shots"));
    m.def("getExpecPauliProd", &getExpecPauliProd, "getExpecPauliProd", py::arg("PauliProdList"));
    m.def("getExpecPauliSum", &getExpecPauliSum, "getExpecPauliSum", py::arg("pauli_type"), py::arg("coeff_type"));

    py::class_<Amplitude>(m, "Amplitude")
        .def(py::init())
        .def_readwrite("reals", &Amplitude::reals)
        .def_readwrite("imags", &Amplitude::imags)
        .def_readwrite("startind", &Amplitude::startind)
        .def_readwrite("numamps", &Amplitude::numamps);

    py::class_<CmdEx>(m, "CmdEx")
        .def(py::init())
        .def_readwrite("amp", &CmdEx::amp);

    py::class_<Cmd>(m, "Cmd")
        .def(py::init())
        .def_readwrite("gate", &Cmd::gate)
        .def_readwrite("targets", &Cmd::targets)
        .def_readwrite("controls", &Cmd::controls)
        .def_readwrite("rotation", &Cmd::rotation)
        .def_readwrite("desc", &Cmd::desc)
        .def_readwrite("inverse", &Cmd::inverse)
        .def_readwrite("cmdex", &Cmd::cmdex);

    py::class_<MeasureResult>(m, "MeasureResult")
        .def(py::init())
        .def_readwrite("id", &MeasureResult::id)
        .def_readwrite("value", &MeasureResult::value);

    py::class_<Outcome>(m, "Outcome")
        .def(py::init())
        .def_readwrite("bitstr", &Outcome::bitstr)
        .def_readwrite("count", &Outcome::count);

    py::class_<Result>(m, "Result")
        .def(py::init())
        .def_readwrite("measureSet", &Result::measureSet)
        .def_readwrite("outcomeSet", &Result::outcomeSet);

    py::class_<PauliProdInfo>(m, "PauliProdInfo")
        .def(py::init())
        .def_readwrite("oper_type", &PauliProdInfo::oper_type)
        .def_readwrite("target", &PauliProdInfo::target);
}