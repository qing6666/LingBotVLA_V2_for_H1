#!/usr/bin/env python3
"""Step 3: URDF -> MJCF 转换。

原理: MuJoCo 内部会把 URDF 先转成 MJCF 再编译。我们利用这一点:
  1. MjModel.from_xml_path() 编译 URDF  → 如果 mesh 路径/惯量/限位有问题, 这里直接报错
  2. mj_saveLastXML() 把编译好的模型回存为 MJCF 文件
生成的 MJCF 保留了 URDF 全部信息(关节限位/阻尼/惯量/mesh),
但还没有执行器/场景 —— 那是 Step 4/6 的事。

输入: urdf/H1_fixed.urdf
输出: mujoco/H1.xml
"""

from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent.parent
URDF = HERE / "urdf/H1_fixed.urdf"
MJCF = HERE / "mujoco/H1.xml"


def main() -> None:
    # ① 编译 URDF (mesh 路径以 URDF 所在目录为基准, 所以 ../meshes 能找到)
    model = mujoco.MjModel.from_xml_path(str(URDF))
    print(f"① URDF 编译成功")
    print(f"   nq={model.nq} (关节数)  nv={model.nv}  nbody={model.nbody}  "
          f"nmesh={model.nmesh}  ngeom={model.ngeom}")

    # 抽查关节限位是否真的进来了 (对照修复脚本的 SPEC)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Left_J4")
    print(f"② 抽查 Left_J4 限位: range={model.jnt_range[jid]} (期望 [0, 2.0944])")
    print(f"   抽查 Left_J4 阻尼: damping={model.dof_damping[model.jnt_dofadr[jid]]} (期望 0.3)")

    # ③ 回存为 MJCF
    MJCF.parent.mkdir(exist_ok=True)
    mujoco.mj_saveLastXML(str(MJCF), model)
    print(f"③ MJCF 已保存 -> {MJCF}")

    # ④ 回读验证 (确保生成的文件自己也能编译 —— round-trip 检查)
    model2 = mujoco.MjModel.from_xml_path(str(MJCF))
    assert model2.nq == model.nq and model2.nbody == model.nbody
    print(f"④ 回读验证通过 (nq={model2.nq}, nbody={model2.nbody} 与编译时一致)")


if __name__ == "__main__":
    main()
