"""
Prints the information for tensorcircuit installation and environment.
"""

import platform
import sys
import numpy


def about() -> None:
    """
    Prints the information for tensorcircuit installation and environment.
    """
    print(f"OS info: {platform.platform(aliased=True)}")
    print(
        f"Python version: {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}"
    )
    print(f"Numpy version: {numpy.__version__}")

    try:
        import scipy

        print(f"Scipy version: {scipy.__version__}")
    except ModuleNotFoundError:
        print(f"Scipy is not installed")
    except Exception as e:
        print(f"Misconfiguration for Scipy: {e}")

    try:
        import pandas

        print(f"Pandas version: {pandas.__version__}")
    except ModuleNotFoundError:
        print(f"Pandas is not installed")
    except Exception as e:
        print(f"Misconfiguration for Pandas: {e}")

    try:
        import tensornetwork as tn

        print(f"TensorNetwork version: {tn.__version__}")
    except ModuleNotFoundError:
        print(f"TensorNetwork is not installed")
    except Exception as e:
        print(f"Misconfiguration for TensorNetwork: {e}")

    try:
        import cotengra

        try:
            print(f"Cotengra version: {cotengra.__version__}")
        except AttributeError:
            print(f"Cotengra: installed")
    except ModuleNotFoundError:
        print(f"Cotengra is not installed")
    except Exception as e:
        print(f"Misconfiguration for Cotengra: {e}")

    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")
        print(f"TensorFlow CUDA infos: {dict(tf.sysconfig.get_build_info())}")
    except ModuleNotFoundError:
        print(f"TensorFlow is not installed")
    except Exception as e:
        print(f"Misconfiguration for TensorFlow: {e}")

    try:
        import jax

        print(f"Jax version: {jax.__version__}")
        try:
            device = jax.devices("gpu")
            print(f"Jax GPU: {device}")
        except RuntimeError:
            print(f"Jax installation doesn't support GPU")
    except ModuleNotFoundError:
        print(f"Jax is not installed")
    except Exception as e:
        print(f"Misconfiguration for Jax: {e}")

    try:
        import jaxlib

        print(f"JaxLib version: {jaxlib.__version__}")
    except ModuleNotFoundError:
        print(f"JaxLib is not installed")
    except Exception as e:
        print(f"Misconfiguration for Jaxlib: {e}")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch GPU support: {torch.cuda.is_available()}")
        print(
            f"PyTorch GPUs: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}"
        )
        if torch.version.cuda is not None:
            print(f"Pytorch cuda version: {torch.version.cuda}")
    except ModuleNotFoundError:
        print(f"PyTorch is not installed")
    except Exception as e:
        print(f"Misconfiguration for Torch: {e}")

    try:
        import cupy

        print(f"Cupy version: {cupy.__version__}")
    except ModuleNotFoundError:
        print(f"Cupy is not installed")
    except Exception as e:
        print(f"Misconfiguration for Cupy: {e}")

    try:
        import qiskit

        print(f"Qiskit version: {qiskit.__version__}")
    except ModuleNotFoundError:
        print(f"Qiskit is not installed")
    except Exception as e:
        print(f"Misconfiguration for Qiskit: {e}")

    try:
        import cirq

        print(f"Cirq version: {cirq.__version__}")
    except ModuleNotFoundError:
        print(f"Cirq is not installed")
    except Exception as e:
        print(f"Misconfiguration for Cirq: {e}")

    from tensorcircuit import __version__

    print(f"TensorCircuit version {__version__}")


def cite(format: str = "bib") -> str:
    """
    Returns the citation information for tensorcircuit.
    Please cite our work if you use the package in your research.

    :param format: format for bib, defaults to "bib"
    :type format: str, optional
    :return: the citation information
    :rtype: str
    """
    if format == "bib":
        return """@article{Zhang_TensorCircuit_2023,
        author = {Zhang, Shi-Xin and Allcock, Jonathan and Wan, Zhou-Quan and Liu, Shuo and Sun, Jiace and Yu, Hao and Yang, Xing-Han and Qiu, Jiezhong and Ye, Zhaofeng and Chen, Yu-Qin and Lee, Chee-Kong and Zheng, Yi-Cong and Jian, Shao-Kai and Yao, Hong and Hsieh, Chang-Yu and Zhang, Shengyu},
        doi = {10.22331/q-2023-02-02-912},
        journal = {Quantum},
        month = feb,
        title = {{TensorCircuit: a Quantum Software Framework for the NISQ Era}},
        volume = {7},
        year = {2023}}"""
    elif format == "aps":
        return """S.-X. Zhang, J. Allcock, Z.-Q. Wan, S. Liu, J. Sun, H. Yu, X.-H. Yang, J. Qiu, Z. Ye, Y.-Q. Chen, C.-K. Lee, Y.-C. Zheng, S.-K. Jian, H. Yao, C.-Y. Hsieh, and S. Zhang, TensorCircuit: a Quantum Software Framework for the NISQ Era, Quantum 7, 912 (2023)."""  # pylint: disable=line-too-long
    raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    about()
