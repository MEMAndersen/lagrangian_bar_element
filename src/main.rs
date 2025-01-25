use std::{collections::HashSet, iter::zip};

use ndarray::{arr2, array, Array1, Array2};
use ndarray_linalg::Solve;
use plotters::prelude::*;

#[derive(Debug, Clone)]
struct Node {
    coord: (f64, f64),
    dofs: (usize, usize),
}

#[derive(Debug)]
struct LagrangianBarElement {
    nodes: (Node, Node),
    e_module: f64,
    area: f64,
}

impl LagrangianBarElement {
    fn length(&self) -> f64 {
        ((self.nodes.1.coord.0 - self.nodes.0.coord.0).powf(2.0)
            + (self.nodes.1.coord.1 - self.nodes.0.coord.1).powf(2.0))
        .sqrt()
    }

    fn dofs(&self) -> [usize; 4] {
        [
            self.nodes.0.dofs.0,
            self.nodes.0.dofs.1,
            self.nodes.1.dofs.0,
            self.nodes.1.dofs.1,
        ]
    }

    fn deformation_interpolation_matrix(&self, x_l: f64) -> Array2<f64> {
        let n1: f64 = x_l / self.length();
        let n0: f64 = 1.0 - &n1;

        array![[n0, 0.0, n1, 0.0], [0.0, n0, 0.0, n1]]
    }

    fn strain_interpolation_matrix(&self) -> Array2<f64> {
        let k: f64 = 1.0 / self.length();

        arr2(&[[-1.0, 0.0, 1.0, 0.0]]) * k
    }

    fn transformation_matrix(&self) -> Array2<f64> {
        let l: f64 = self.length();
        let nx: f64 = (self.nodes.1.coord.0 - self.nodes.0.coord.0) / l;
        let ny: f64 = (self.nodes.1.coord.1 - self.nodes.0.coord.1) / l;

        arr2(&[
            [nx, ny, 0.0, 0.0],
            [-ny, nx, 0.0, 0.0],
            [0.0, 0.0, nx, ny],
            [0.0, 0.0, -nx, ny],
        ])
    }

    fn axial_force(&self, ve: &Vec<f64>) -> Array2<f64> {
        let ve: Array2<f64> = Array2::from_shape_vec((4, 1), ve.clone()).unwrap();
        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let (gx, _, g): (Array2<f64>, Array2<f64>, Array2<f64>) = self.g_parts();
        let local_ve: Array2<f64> = transformation_matrix.dot(&ve);
        let local_strain_linear: Array2<f64> = gx.dot(&local_ve);
        let local_strain_non_linear: Array2<f64> = 0.5 * (&local_ve.t().dot(&g)).dot(&local_ve);

        self.area * self.e_module * (local_strain_linear + local_strain_non_linear)
    }

    fn g_parts(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let gx = self.strain_interpolation_matrix();
        let gy = -gx.clone();
        let g = &gx.t().dot(&gx) + &gy.t().dot(&gy);
        (gx, gy, g)
    }

    fn tangent_stiffness_matrix(&self, ve: &Vec<f64>) -> Array2<f64> {
        // ve should be of length 4

        let l: f64 = self.length();
        let ea: f64 = self.area * self.e_module;
        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let (gx, _, g): (Array2<f64>, Array2<f64>, Array2<f64>) = self.g_parts();

        let axial_force: Array2<f64> = self.axial_force(&ve);

        let ve: Array2<f64> = Array2::from_shape_vec((4, 1), ve.clone()).unwrap();
        let local_ve: &Array2<f64> = &transformation_matrix.dot(&ve);

        let k_sigma: Array2<f64> = l * axial_force * &g;

        let bl: Array2<f64> = gx + local_ve.t().dot(&g);
        let k_0: Array2<f64> = l * ea * bl.dot(&bl.t());

        let kl: Array2<f64> = k_sigma + k_0;

        kl * &transformation_matrix.t().dot(&transformation_matrix)
    }

    fn tangent_stiffness_matrix_as_dof_index(&self, ve: &Vec<f64>) -> Vec<((usize, usize), f64)> {
        let k: Array2<f64> = self.tangent_stiffness_matrix(&ve);

        let dofs: [usize; 4] = self.dofs();

        k.indexed_iter()
            .map(|((row, col), val)| ((dofs[row], dofs[col]), *val))
            .collect()
    }

    fn force_vector(&self, ve: &Vec<f64>) -> Array2<f64> {
        let axial_force: Array2<f64> = self.axial_force(&ve);
        let ve: Array2<f64> = Array2::from_shape_vec((4, 1), ve.clone()).unwrap();

        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let l = self.length();
        let (gx, _, g): (Array2<f64>, _, Array2<f64>) = self.g_parts();
        let local_ve: &Array2<f64> = &transformation_matrix.dot(&ve);

        let element_force_vector: Array2<f64> = l * &axial_force * (&gx.t() + g.dot(local_ve));

        let global_force_vector: Array2<f64> = transformation_matrix.dot(&element_force_vector);

        global_force_vector
    }

    fn draw_topology_data(&self) -> [(f64, f64); 2] {
        [
            (self.nodes.0.coord.0, self.nodes.0.coord.1),
            (self.nodes.1.coord.0, self.nodes.1.coord.1),
        ]
    }

    fn draw_topology(&self) -> LineSeries<BitMapBackend<'_>, (f64, f64)> {
        LineSeries::new(self.draw_topology_data().map(|point| point), BLACK.filled()).point_size(4)
    }
}

#[derive(Debug)]
struct Load {
    node: Node,
    value: (f64, f64),
}

#[derive(Debug)]
struct Support {
    node: Node,
    value: (bool, bool),
}

#[derive(Debug)]
enum BoundaryCondition {
    Load(Load),
    Support(Support),
}

#[derive(Debug)]
struct Structure {
    nodes: Vec<Node>,
    elements: Vec<LagrangianBarElement>,
    loads: Vec<Load>,
    supports: Vec<Support>,
    nno: usize,
    ndof: usize,
    nel: usize,
}

impl Structure {
    fn new(
        nodes: Vec<Node>,
        elements: Vec<LagrangianBarElement>,
        loads: Vec<Load>,
        supports: Vec<Support>,
    ) -> Self {
        let nno = nodes.len();
        let ndof = nno * 2;
        let nel = elements.len();

        Structure {
            nodes,
            elements,
            loads,
            supports,
            nno,
            ndof,
            nel,
        }
    }

    fn plot_topology(&self) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("out/topology.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Topology", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-1f64..1f64, -0.1f64..1f64)?;

        chart.configure_mesh().draw()?;

        for element in &self.elements {
            chart.draw_series(element.draw_topology()).unwrap();
        }

        root.present()?;

        Ok(())
    }

    fn get_supported_dofs(&self) -> Vec<usize> {
        let mut supported_dofs_set: HashSet<usize> = HashSet::new();

        for support in &self.supports {
            if support.value.0 {
                supported_dofs_set.insert(support.node.dofs.0);
            }
            if support.value.1 {
                supported_dofs_set.insert(support.node.dofs.1);
            }
        }
        let mut supported_dofs_vec: Vec<usize> =
            supported_dofs_set.into_iter().collect::<Vec<usize>>();
        supported_dofs_vec.sort();
        supported_dofs_vec
    }

    fn get_step_load_vector(&self) -> Array1<f64> {
        let mut load_vector: Array1<f64> = Array1::zeros(self.ndof);
        self.loads.iter().for_each(|load| {
            load_vector[load.node.dofs.0] += load.value.0;
            load_vector[load.node.dofs.1] += load.value.1;
        });
        load_vector
    }

    fn system_tangent_stiffness_matrix(&self, gve: &Array1<f64>) -> Array2<f64> {
        let mut matrix: Array2<f64> = Array2::zeros((self.nno * 2, self.nno * 2));

        for element in &self.elements {
            let ve: Vec<f64> = element.dofs().iter().map(|dof: &usize| gve[*dof]).collect();

            element
                .tangent_stiffness_matrix_as_dof_index(&ve)
                .iter()
                .for_each(|((row, col), val)| matrix[[*row, *col]] += val);
        }
        matrix
    }

    fn system_force_vector(&self, gve: &Array1<f64>) -> Array1<f64> {
        let mut vector: Array1<f64> = Array1::zeros(self.ndof);

        for element in &self.elements {
            let ve: Vec<f64> = element.dofs().iter().map(|dof| gve[*dof]).collect();
            for (dof, force) in zip(element.dofs(), element.force_vector(&ve)) {
                vector[dof] += force
            }
        }

        vector
    }

    fn solve(&self, nstep: usize, nit: usize, umax: f64, tol: f64) {
        // Initialize
        let mut deformation_vector: Array1<f64> = Array1::zeros(self.ndof);
        let mut load_vector: Array1<f64> = Array1::zeros(self.ndof);
        let step_load_vector: Array1<f64> = self.get_step_load_vector();
        let mut delta_load_vector: Array1<f64> = Array1::zeros(self.ndof);

        let mut residual_load_vector: Array1<f64> = Array1::zeros(self.ndof);

        let mut delta_deformation_vector_prev: Array1<f64> = Array1::zeros(self.ndof);
        let mut delta_load_vector_prev: Array1<f64> = Array1::zeros(self.ndof);

        let supported_dofs = self.get_supported_dofs();

        for step in 1..nstep {
            // Setup displacement and load vectors

            // Previous steps
            let deformation_vector_prev: Array1<f64> = deformation_vector.clone();
            let load_vector_prev: Array1<f64> = load_vector.clone();

            // update for current step
            delta_load_vector = step_load_vector.clone();
            load_vector = load_vector + &delta_load_vector;
            residual_load_vector = residual_load_vector + &delta_load_vector;

            // Start iteration
            let j = 0;
            let delta_deformation_vector: Array1<f64> = Array1::ones(self.ndof);
            let intial_iteration_energy: f64 = 0.0;

            while (j < nit)
                & (tol * intial_iteration_energy
                    < (&residual_load_vector.dot(&delta_deformation_vector)).abs())
            {
                let mut tangent_stiffness_matrix: Array2<f64> =
                    self.system_tangent_stiffness_matrix(&load_vector);

                // Supports
                apply_stiffness_to_supports(&supported_dofs, &mut tangent_stiffness_matrix);

                // Solve for residual
                delta_deformation_vector = tangent_stiffness_matrix.solve_into(r)
            }
        }
    }
}

fn apply_stiffness_to_supports(
    supported_dofs: &Vec<usize>,
    tangent_stiffness_matrix: &mut Array2<f64>,
) {
    let kmax: f64 = tangent_stiffness_matrix
        .diag()
        .iter()
        .cloned()
        .fold(0.0, |max: f64, x: f64| max.max(x))
        * 1e9;

    tangent_stiffness_matrix
        .diag_mut()
        .iter_mut()
        .enumerate()
        .for_each(|(i, diag_view)| {
            if supported_dofs.contains(&i) {
                *diag_view += kmax
            }
        });
}

fn main() {
    const A_DIM: f64 = 3.0;
    const HEIGHT: f64 = 0.2;
    let l0: f64 = ((2.0 * A_DIM).powf(2.0) + (HEIGHT).powf(2.0)).sqrt();

    const E: f64 = 2.0;
    const AREA: f64 = 0.25;

    let e_norm: f64 = (2.0 * A_DIM - l0) / l0;
    let pn: f64 = E * AREA * (e_norm);

    let nodes: Vec<Node> = vec![
        Node {
            coord: (-A_DIM, 0.0),
            dofs: (0, 1),
        },
        Node {
            coord: (0.0, HEIGHT),
            dofs: (2, 3),
        },
        Node {
            coord: (A_DIM, 0.0),
            dofs: (4, 5),
        },
    ];

    let structure = Structure::new(
        nodes.clone(),
        vec![
            LagrangianBarElement {
                nodes: (nodes[0].clone(), nodes[1].clone()),
                e_module: E,
                area: AREA,
            },
            LagrangianBarElement {
                nodes: (nodes[1].clone(), nodes[2].clone()),
                e_module: E,
                area: AREA,
            },
        ],
        vec![Load {
            node: nodes[1].clone(),
            value: (0.0, -0.01 * pn),
        }],
        vec![
            Support {
                node: nodes[0].clone(),
                value: (true, true),
            },
            Support {
                node: nodes[2].clone(),
                value: (true, true),
            },
        ],
    );

    let test_vec0: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0];
    let test_vec1: Vec<f64> = vec![0.0, 1.0, 0.0, 0.0];

    println!("{}\n", structure.elements[0].force_vector(&test_vec0));
    println!("{}\n", structure.elements[1].force_vector(&test_vec1));

    println!(
        "{}\n",
        structure.elements[0].tangent_stiffness_matrix(&test_vec0)
    );
    println!(
        "{}\n",
        structure.elements[1].tangent_stiffness_matrix(&test_vec1)
    );

    println!(
        "{:?}\n",
        structure.elements[0].tangent_stiffness_matrix_as_dof_index(&test_vec0)
    );
    println!(
        "{:?}\n",
        structure.elements[1].tangent_stiffness_matrix_as_dof_index(&test_vec1)
    );

    let test_vec99: Vec<f64> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let gve = Array1::from_vec(test_vec99);

    println!("{:?}\n", structure.system_tangent_stiffness_matrix(&gve));

    println!("{:?}\n", structure.system_force_vector(&gve));

    let _ = structure.plot_topology();

    structure.solve(200, 10, 0.5, 1e-4);
}
