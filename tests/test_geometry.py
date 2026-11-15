from relaxfree.geometry import displacement_from_ideal

HEADER = "_cell_length_a   10.0\nloop_\n"


def _cif(rows):
    return HEADER + "".join(f"  Si  Si{i}  1  {x:.8f}  {y:.8f}  {z:.8f}  1\n" for i, (x, y, z) in enumerate(rows))


def test_ideal_sites_have_zero_displacement():
    n, d = displacement_from_ideal(_cif([(0.0, 0.25, 0.5), (0.125, 0.375, 0.875), (0.0625, 0.5625, 0.9375)]))
    assert n == 0 and d < 1e-9


def test_displaced_atom_is_detected():
    n, d = displacement_from_ideal(_cif([(0.0, 0.25, 0.5), (0.130, 0.375, 0.875)]))
    assert n == 1
    assert abs(d - 0.05) < 1e-6


def _cell_cif(a, rows):
    head = (f"_cell_length_a {a}\n_cell_length_b {a}\n_cell_length_c {a}\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\nloop_\n"
            " _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n"
            " _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n")
    return head + "".join(f"  Si  Si{i}  1  {x}  {y}  {z}  1\n" for i, (x, y, z) in enumerate(rows))


def test_displacement_stats(tmp_path):
    from relaxfree.structures import displacement_stats
    a = tmp_path / "a.cif"
    b = tmp_path / "b.cif"
    a.write_text(_cell_cif(10.0, [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.99, 0.0, 0.0)]))
    b.write_text(_cell_cif(10.0, [(0.0, 0.0, 0.0), (0.52, 0.5, 0.5), (0.01, 0.0, 0.0)]))
    st = displacement_stats(a, b)
    assert st["n_atoms"] == 3
    assert abs(st["u_max"] - 0.2) < 1e-9  # both moved atoms shift by 0.2 A, one across the boundary
    assert abs(st["frac_moved"] - 2 / 3) < 1e-9
