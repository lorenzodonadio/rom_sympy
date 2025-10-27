import sympy as sm
import sympy.physics.mechanics as me


class SemiFlexBody:
    def __init__(
        self,
        ref: me.ReferenceFrame,
        O: me.Point,
        name: str,
        xcm=0,
        ycm=0,
        zcm=0,
        dof_x=0,
        dof_y=0,
        dof_z=0,
        prefix=None,
        var_start=1,
        elastic_forces=True,
        damp_forces=True,
        uniform_k=False,
        uniform_m=False,
        uniform_damp=False,
    ):

        self.name = name
        self.prefix = prefix or name[:2].upper()
        self.dof_x = dof_x
        self.dof_y = dof_y
        self.dof_z = dof_z
        self.var_start = var_start
        self._is_xflex = dof_x > 0
        self._is_yflex = dof_y > 0
        self._is_zflex = dof_z > 0
        self.O = O
        self.ref = ref
        self.loads = []
        assert (
            self._is_xflex + self._is_yflex + self._is_zflex < 3
        ), "Maximum of 2 coordinates can have DOFs"

        final_q_num = dof_x + dof_y + dof_z + var_start
        self.q = me.dynamicsymbols(f"q{var_start}:{final_q_num}")
        self.u = me.dynamicsymbols(f"u{var_start}:{final_q_num}")
        # Mass
        if uniform_m:
            if isinstance(uniform_m, sm.Symbol):
                m_sym = uniform_m
            else:
                m_sym = sm.symbols(f"m_{self.prefix}")
            self._mx = (m_sym,) * self.dof_x
            self._my = (m_sym,) * self.dof_y
            self._mz = (m_sym,) * self.dof_z
        else:
            self._mx = sm.symbols(f"m_x{self.prefix}_1:{self.dof_x+1}")
            self._my = sm.symbols(f"m_y{self.prefix}_1:{self.dof_y+1}")
            self._mz = sm.symbols(f"m_z{self.prefix}_1:{self.dof_z+1}")

        self.m = self._mx + self._my + self._mz

        # Stiffness
        if uniform_k:
            if isinstance(uniform_k, sm.Symbol):
                k_sym = uniform_k
            else:
                k_sym = sm.symbols(f"k_{self.prefix}")

            self._kx = (k_sym,) * self.dof_x
            self._ky = (k_sym,) * self.dof_y
            self._kz = (k_sym,) * self.dof_z
        else:
            self._kx = sm.symbols(f"k_x{self.prefix}_1:{self.dof_x+1}")
            self._ky = sm.symbols(f"k_y{self.prefix}_1:{self.dof_y+1}")
            self._kz = sm.symbols(f"k_z{self.prefix}_1:{self.dof_z+1}")

        self.k = self._kx + self._ky + self._kz

        # Damping
        if uniform_damp:
            if isinstance(uniform_damp, sm.Symbol):
                c_sym = uniform_damp
            else:
                c_sym = sm.symbols(f"c_{self.prefix}")

            self._cx = (c_sym,) * self.dof_x
            self._cy = (c_sym,) * self.dof_y
            self._cz = (c_sym,) * self.dof_z
        else:
            self._cx = sm.symbols(f"c_x{self.prefix}_1:{self.dof_x+1}")
            self._cy = sm.symbols(f"c_y{self.prefix}_1:{self.dof_y+1}")
            self._cz = sm.symbols(f"c_z{self.prefix}_1:{self.dof_z+1}")

        self.c = self._cx + self._cy + self._cz

        self._rx = sm.symbols(f"r_x{self.prefix}_1:{self.dof_x+1}")
        self._ry = sm.symbols(f"r_y{self.prefix}_1:{self.dof_y+1}")
        self._rz = sm.symbols(f"r_z{self.prefix}_1:{self.dof_z+1}")

        if self._is_xflex:
            self.theta_x = sm.Matrix(self._qx()).dot(sm.Matrix(self._rx))

        if self._is_yflex:
            self.theta_y = sm.Matrix(self._qy()).dot(sm.Matrix(self._ry))

        if self._is_zflex:
            self.theta_z = sm.Matrix(self._qz()).dot(sm.Matrix(self._rz))

        self.locate(xcm, ycm, zcm)

        if elastic_forces:
            self.add_elastic_loads()
        if damp_forces:
            self.add_damp_loads()

        self.free_symbols = (
            set(self.m) | set(self.k) | set(self.c) | set(self._rx + self._ry + self._rz)
        )

    def locate(self, x, y, z):

        cm_v = x * self.ref.x + y * self.ref.y + z * self.ref.z
        self.masscenter = me.Point(f"cm_{self.prefix}")
        self.masscenter.set_pos(self.O, cm_v)
        R = self.ref
        _op_x = [
            self.O.locatenew(f"op_x{self.prefix}_{i+1}", cm_v + q * R.x)
            for i, q in enumerate(self._qx())
        ]
        _op_y = [
            self.O.locatenew(f"op_y{self.prefix}_{i+1}", cm_v + q * R.y)
            for i, q in enumerate(self._qy())
        ]
        _op_z = [
            self.O.locatenew(f"op_z{self.prefix}_{i+1}", cm_v + q * R.z)
            for i, q in enumerate(self._qz())
        ]

        pn = lambda n, i: f"P{self.prefix}{n}_{i+1}"

        self.P_x = tuple(
            [me.Particle(pn("x", i), op, m) for i, (op, m) in enumerate(zip(_op_x, self._mx))]
        )
        self.P_y = tuple(
            [me.Particle(pn("y", i), op, m) for i, (op, m) in enumerate(zip(_op_y, self._my))]
        )
        self.P_z = tuple(
            [me.Particle(pn("z", i), op, m) for i, (op, m) in enumerate(zip(_op_z, self._mz))]
        )

    def add_elastic_loads(self):
        for i, P in enumerate(self.P_x):
            q = self._qx()
            self.loads.append((P.masscenter, -self._kx[i] * q[i] * self.ref.x))
        for i, P in enumerate(self.P_y):
            q = self._qy()
            self.loads.append((P.masscenter, -self._ky[i] * q[i] * self.ref.y))
        for i, P in enumerate(self.P_z):
            q = self._qz()
            self.loads.append((P.masscenter, -self._kz[i] * q[i] * self.ref.z))

    def add_damp_loads(self):
        for i, P in enumerate(self.P_x):
            u = self._ux()
            self.loads.append((P.masscenter, -self._cx[i] * u[i] * self.ref.x))
        for i, P in enumerate(self.P_y):
            u = self._uy()
            self.loads.append((P.masscenter, -self._cy[i] * u[i] * self.ref.y))
        for i, P in enumerate(self.P_z):
            u = self._uz()
            self.loads.append((P.masscenter, -self._cz[i] * u[i] * self.ref.z))

    def get_bodies(self):
        return [*self.P_x, *self.P_y, *self.P_z]

    def get_subs_dict(self, value=0.0):
        """Returns a dictionary mapping all unique symbols to value (default 0), and a symbol map so that the user can modify that."""
        return {sym: value for sym in self.free_symbols}, self.get_symbol_map()

    def get_symbol_map(self):
        """Returns a dict mapping symbol names (as strings) to symbol objects."""
        return {str(sym): sym for sym in self.free_symbols}

    def _qx(self):
        """get qs in x direction"""
        return self.q[: self.dof_x]  # type: ignore

    def _qy(self):
        """get qs in y direction"""

        return self.q[self.dof_x : self.dof_y + self.dof_x]  # type: ignore

    def _qz(self):
        """get qs in z direction"""
        return self.q[self.dof_y + self.dof_x : self.dof_y + self.dof_x + self.dof_z]  # type: ignore

    def _ux(self):
        """get us in x direction"""
        return self.u[: self.dof_x]  # type: ignore

    def _uy(self):
        """get us in y direction"""

        return self.u[self.dof_x : self.dof_y + self.dof_x]  # type: ignore

    def _uz(self):
        """get us in z direction"""
        return self.u[self.dof_y + self.dof_x : self.dof_y + self.dof_x + self.dof_z]  # type: ignore
