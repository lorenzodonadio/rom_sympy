import re
import sympy as sm


def add_line_under_subroutine(file_path, text=""):
    """Insert 'text' after every complete multi-line subroutine declaration in a Fortran file."""

    with open(file_path, "r") as f:
        lines = f.readlines()
    look_for_closing_subroutine = False
    idx_to_add = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if look_for_closing_subroutine and ")" in stripped:
            idx_to_add.append(i + 1)
            look_for_closing_subroutine = False
            continue
        if stripped.startswith("subroutine "):
            if ")" in stripped:
                idx_to_add.append(i + 1)
            else:
                look_for_closing_subroutine = True

    indent_match = re.match(r"^(\s*)", lines[idx_to_add[0]])
    indent = indent_match.group(1) if indent_match else ""
    for j, i in enumerate(idx_to_add):
        lines = lines[: i + j] + [f"{indent}{text}\n"] + lines[i + j :]
    with open(file_path, "w") as f:
        f.writelines(lines)
    print(f"✅ Modified file saved: {file_path}")


def add_txt_after_sep(s: str, txt=" &\n", min_len=20, sep=","):
    start = 0
    ltxt = len(txt)
    while len(s) > start + min_len:
        if sep not in s[start + min_len :]:
            return s
        offset = s[start + min_len :].index(sep) + 1
        idx = start + min_len + offset
        s = s[:idx] + txt + s[idx:]
        start += min_len + ltxt  # 3 is len(' &\n')
    return s


def create_data_module(module_name: str, namelist_name: str, vars: list[str], output_path="."):
    namelist = f"  namelist /{namelist_name}/ " + ", ".join(vars)
    namelist = add_txt_after_sep(namelist, " &\n  ", 50)  # just break lines so it looks nice

    lines = [
        f"module {module_name}",
        "  implicit none",
        "",
        "  ! Real parameters (double precision)",
    ]

    lines = lines + [f"  real(8) :: {var} = 0.0d0" for var in vars]
    lines.append("")
    lines.append(namelist)
    lines.append("")
    lines.append("contains")
    lines.append("")

    subroutine_init = [
        f"   subroutine init_{module_name}(nml_filename)",
        "",
        "      character(len=256),intent(in) :: nml_filename",
        "      open(unit=11, file=nml_filename, status='old', action='read')",
        f"      read(11,nml={namelist_name})",
        "      close(11)",
        "",
        f"   end subroutine init_{module_name}",
    ]

    lines = lines + subroutine_init
    lines.append(f"end module {module_name}")

    # Write to file
    fname = output_path + f"/{module_name}.f90"
    with open(fname, "w+") as f:
        f.write("\n".join(lines))

    print(f"✅ {fname} generated!")


def create_init_namelist(
    name: str,
    vars: list[str],
    output_path=".",
    verbose=True,
):
    datalines = [f"{var} = 0.0d0" for var in vars]
    lines = [f"&{name}"] + datalines + ["/\n"]

    fname = output_path + f"/{name.lower()}.nml"

    with open(fname, "w+") as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"✅ {fname} generated!")


def dict_to_namelist(
    name: str,
    data: dict[str, float],
    output_path=".",
    module_vars: list[str] | None = None,
    verbose=True,
):
    if module_vars is not None:
        datalines = [f"{k} = {v:.5e}" for k, v in data.items() if k in module_vars]
    else:
        datalines = [f"{k} = {v:.5e}" for k, v in data.items()]
    lines = [f"&{name}"] + datalines + ["/\n"]

    fname = output_path + f"/{name.lower()}.nml"

    with open(fname, "w+") as f:
        f.write("\n".join(lines))
    if verbose:
        print(f"✅ {fname} generated!")


def d2s(km_results):
    """returns a subs dict converting the dynamic symbols in km_results['coordinates'] and km_results['speeds'] into
    normal symbols q1..n u1..n so that converting the code to fortran works well"""
    ncoords = len(km_results["coordinates"]) + 1
    nspeeds = len(km_results["speeds"]) + 1

    qsym = sm.symbols(f"q1:{ncoords}")
    usym = sm.symbols(f"u1:{nspeeds}")

    dyn_to_sym = {}  # replace q1(t) by q1, make t dissapear

    for d, s in zip(km_results["coordinates"], qsym):
        dyn_to_sym[d] = s
    for d, s in zip(km_results["speeds"], usym):
        dyn_to_sym[d] = s
    return dyn_to_sym
