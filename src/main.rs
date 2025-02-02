use std::fmt::format;
use std::{collections::HashSet, iter::zip};

use ndarray::{arr1, arr2, array, Array1, Array2};
use ndarray_linalg::Solve;
use plotters::prelude::*;

use std::fs::File;
use std::io::prelude::*;

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

    fn strain_interpolation_matrix(&self) -> (Array2<f64>, Array2<f64>) {
        let k: f64 = 1.0 / self.length();

        (
            arr2(&[[-1.0, 0.0, 1.0, 0.0]]) * k,
            arr2(&[[0.0, -1.0, 0.0, 1.0]]) * k,
        )
    }

    fn transformation_matrix(&self) -> Array2<f64> {
        let l: f64 = self.length();
        let nx: f64 = (self.nodes.1.coord.0 - self.nodes.0.coord.0) / l;
        let ny: f64 = (self.nodes.1.coord.1 - self.nodes.0.coord.1) / l;

        arr2(&[
            [nx, ny, 0.0, 0.0],
            [-ny, nx, 0.0, 0.0],
            [0.0, 0.0, nx, ny],
            [0.0, 0.0, -ny, nx],
        ])
    }

    fn axial_force(&self, ve: &Vec<f64>) -> Array2<f64> {
        let ve: Array2<f64> = Array2::from_shape_vec((4, 1), ve.clone()).unwrap();
        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let (gx, _, g): (Array2<f64>, Array2<f64>, Array2<f64>) = self.g_parts();

        // Calculate strain
        let local_ve: Array2<f64> = transformation_matrix.dot(&ve);
        let local_strain_linear: Array2<f64> = gx.dot(&local_ve);
        let local_strain_non_linear: Array2<f64> = 0.5 * (&local_ve.t().dot(&g)).dot(&local_ve);

        self.area * self.e_module * (local_strain_linear + local_strain_non_linear)
    }

    fn g_parts(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let (gx, gy) = self.strain_interpolation_matrix();
        let g = &gx.t().dot(&gx) + &gy.t().dot(&gy);
        (gx, gy, g)
    }

    fn tangent_stiffness_matrix(&self, ve: &Vec<f64>) -> Array2<f64> {
        // ve should be of length 4

        // Scalars
        let l: f64 = self.length();
        let ea: Array2<f64> = arr2(&[[self.area * self.e_module]]);

        // Transformation and strain interpolationi matrices
        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let (gx, _, g): (Array2<f64>, Array2<f64>, Array2<f64>) = self.g_parts();

        let axial_force: Array2<f64> = self.axial_force(&ve);

        let ve: Array2<f64> = Array2::from_shape_vec((4, 1), ve.clone()).unwrap();
        let local_ve: &Array2<f64> = &transformation_matrix.dot(&ve);

        let k_sigma: Array2<f64> = l * axial_force * &g;

        let bl: Array2<f64> = gx + local_ve.t().dot(&g);
        let k_0: Array2<f64> = l * bl.t().dot(&ea).dot(&bl);

        let kl: Array2<f64> = k_sigma + k_0;

        transformation_matrix
            .t()
            .dot(&kl)
            .dot(&transformation_matrix)
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

        let global_force_vector: Array2<f64> = transformation_matrix.t().dot(&element_force_vector);

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

    fn draw_deformed_data(&self, ve: &Vec<f64>) -> [(f64, f64); 2] {
        [
            (self.nodes.0.coord.0 + ve[0], self.nodes.0.coord.1 + ve[1]),
            (self.nodes.1.coord.0 + ve[2], self.nodes.1.coord.1 + ve[3]),
        ]
    }

    fn draw_deformed(&self, ve: &Vec<f64>) -> LineSeries<BitMapBackend<'_>, (f64, f64)> {
        LineSeries::new(
            self.draw_deformed_data(ve).map(|point| point),
            BLACK.filled(),
        )
        .point_size(4)
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

    fn get_plot_bounds(
        &self,
        x_padding: (f64, f64),
        y_padding: (f64, f64),
    ) -> ((f64, f64), (f64, f64)) {
        let xs: Vec<f64> = self.nodes.iter().map(|node| node.coord.0).collect();

        let ys: Vec<f64> = self.nodes.iter().map(|node| node.coord.1).collect();

        let xs_min = xs.iter().cloned().fold(0.0, |a1: f64, a2: f64| a1.min(a2));
        let xs_max = xs.iter().cloned().fold(0.0, |a1: f64, a2: f64| a1.max(a2));
        let ys_min = ys.iter().cloned().fold(0.0, |a1: f64, a2: f64| a1.min(a2));
        let ys_max = ys.iter().cloned().fold(0.0, |a1: f64, a2: f64| a1.max(a2));

        (
            (xs_min - x_padding.0, xs_max - x_padding.1),
            (ys_min - y_padding.0, ys_max - y_padding.1),
        )
    }

    fn plot_topology(&self) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("out/topology.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE)?;

        let ((x_min, x_max), (y_min, y_max)) = self.get_plot_bounds((0.0, 0.0), (0.0, 0.0));

        let mut chart = ChartBuilder::on(&root)
            .caption("Topology", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

        chart.configure_mesh().draw()?;

        for element in &self.elements {
            chart.draw_series(element.draw_topology()).unwrap();
        }

        root.present()?;

        Ok(())
    }

    fn plot_deformed(
        &self,
        gve: &Array1<f64>,
        step: usize,
        plot_bounds: ((f64, f64), (f64, f64)),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = format!("out/deformed_{:3}.png", step);
        let root = BitMapBackend::new(&path, (640, 480)).into_drawing_area();

        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Topology", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                plot_bounds.0 .0..plot_bounds.0 .1,
                plot_bounds.1 .0..plot_bounds.1 .1,
            )?;

        chart.configure_mesh().draw()?;

        for element in &self.elements {
            let ve: Vec<f64> = element.dofs().iter().map(|dof| gve[*dof]).collect();
            chart.draw_series(element.draw_deformed(&ve)).unwrap();
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
        let mut force_vector: Array1<f64> = Array1::zeros(self.ndof);

        for element in &self.elements {
            let ve: Vec<f64> = element.dofs().iter().map(|dof| gve[*dof]).collect();
            for (dof, force) in zip(element.dofs(), element.force_vector(&ve)) {
                force_vector[dof] += force
            }
        }

        force_vector
    }

    fn solve(&self, nstep: usize, nit: usize, umax: f64, tol: f64) {
        // Prepare output file
        let mut output_file = File::create("out/output.txt").expect("Unable to create file");

        // Initialize
        let mut deformation_vector: Array1<f64> = Array1::zeros(self.ndof);
        let mut load_vector: Array1<f64> = Array1::zeros(self.ndof);
        let step_load_vector: Array1<f64> = self.get_step_load_vector();

        let mut residual: Array1<f64> = Array1::zeros(self.ndof);

        let plot_bounds = self.get_plot_bounds((0.1, 0.1), (8.0, 0.1));

        // let mut delta_deformation_vector_prev: Array1<f64> = Array1::zeros(self.ndof);
        // let mut delta_load_vector_prev: Array1<f64> = Array1::zeros(self.ndof);

        let supported_dofs = self.get_supported_dofs();

        for step in 0..nstep {
            // Previous step
            let prev_solve_state = SolveState {
                deformation_vector: deformation_vector.to_owned(),
                load_vector: load_vector.to_owned(),
            };

            // update for current step
            let mut delta_load_vector: Array1<f64> = step_load_vector.clone();
            load_vector = &load_vector + &delta_load_vector;
            residual = residual + &delta_load_vector;

            // Start iteration
            let mut j = 0;
            let delta_deformation: Array1<f64> = Array1::ones(self.ndof);
            let mut intial_iteration_energy: f64 = 0.0;

            while (j < nit)
                & (tol * intial_iteration_energy < (&residual.t().dot(&delta_deformation)).abs())
            {
                let mut tangent_stiffness_matrix: Array2<f64> =
                    self.system_tangent_stiffness_matrix(&deformation_vector);

                // Supports
                apply_stiffness_to_supports(&supported_dofs, &mut tangent_stiffness_matrix);

                // Solve for residual
                let delta_deformation_vector: Array1<f64> =
                    tangent_stiffness_matrix.solve(&residual).unwrap();

                println!("delta_deformation_vector = {}", delta_deformation_vector);

                if (step == 0) & (j == 0) {
                    intial_iteration_energy = delta_deformation_vector.t().dot(&residual);
                    println!("intial_iteration_energy = {}", intial_iteration_energy)
                }

                // Update deformations
                deformation_vector += &delta_deformation_vector;

                // Length control
                let step_deformation: Array1<f64> =
                    &deformation_vector - &prev_solve_state.deformation_vector;
                let squared_total_deformation_length = &step_deformation.t().dot(&step_deformation);

                println!(
                    "squared_total_deformation_length={}",
                    squared_total_deformation_length
                );

                if *squared_total_deformation_length > (&umax * &umax) {
                    let length_factor = (umax * umax) / squared_total_deformation_length;
                    deformation_vector =
                        &prev_solve_state.deformation_vector + length_factor * &step_deformation;
                    delta_load_vector = length_factor * delta_load_vector;
                }

                // Direction control

                // Update load vector after controls
                load_vector = &prev_solve_state.load_vector + &delta_load_vector;
                println!("load_vector = {}", load_vector);

                // Setup force vector
                let force_vector = self.system_force_vector(&deformation_vector);
                println!("force_vector = {}", force_vector);

                //residual
                residual = &load_vector - &force_vector;
                for dof in &supported_dofs {
                    residual[*dof] = 0.0
                }
                println!("residual = {}", residual);

                j += 1;
                println!(
                    "{}: {} < {}",
                    j,
                    tol * intial_iteration_energy,
                    (&residual.t().dot(&step_deformation)).abs()
                );
            }
            let data = format!("{}, {}, {}", -deformation_vector[7], -load_vector[7], nit);
            println!("{}", &data);
            writeln!(&mut output_file, "{}", data).expect("Unable to read line");

            if step % 20 == 0 {
                self.plot_deformed(&deformation_vector, step, plot_bounds)
                    .unwrap();
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

struct SolveState {
    deformation_vector: Array1<f64>,
    load_vector: Array1<f64>,
}

fn main() {
    const A_DIM: f64 = 4.0;
    const HEIGHT: f64 = 4.0;
    const H1: f64 = 0.1;
    const H2: f64 = 0.0;
    let l0: f64 = ((4.0 * A_DIM).powf(2.0) + (HEIGHT).powf(2.0)).sqrt();

    const E: f64 = 210e9;
    const AREA: f64 = 0.1;

    let en: f64 = (4.0 * A_DIM - l0) / l0;
    let pn: f64 = (E * AREA * en).abs();
    let p: f64 = 0.005 * pn;

    let nodes: Vec<Node> = vec![
        Node {
            coord: (0.0, 0.0),
            dofs: (0, 1),
        },
        Node {
            coord: (
                2.0 * A_DIM + 0.5 * H1 * A_DIM / l0 - H2 * A_DIM / l0,
                0.5 * A_DIM - 0.5 * H1 * 4.0 * A_DIM / l0 + H2 * 4.0 * A_DIM / l0,
            ),
            dofs: (2, 3),
        },
        Node {
            coord: (
                2.0 * A_DIM - 0.5 * H1 * A_DIM / l0 - H2 * A_DIM / l0,
                0.5 * A_DIM + 0.5 * H1 * 4.0 * A_DIM / l0 + H2 * 4.0 * A_DIM / l0,
            ),
            dofs: (4, 5),
        },
        Node {
            coord: (4.0 * A_DIM, HEIGHT),
            dofs: (6, 7),
        },
    ];

    println!("{:?}", nodes);

    let structure = Structure::new(
        nodes.clone(),
        vec![
            LagrangianBarElement {
                nodes: (nodes[0].clone(), nodes[1].clone()),
                e_module: E,
                area: AREA,
            },
            LagrangianBarElement {
                nodes: (nodes[0].clone(), nodes[2].clone()),
                e_module: E,
                area: AREA,
            },
            LagrangianBarElement {
                nodes: (nodes[1].clone(), nodes[3].clone()),
                e_module: E,
                area: AREA,
            },
            LagrangianBarElement {
                nodes: (nodes[2].clone(), nodes[3].clone()),
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
            node: nodes[3].clone(),
            value: (0.0, -p),
        }],
        vec![
            Support {
                node: nodes[0].clone(),
                value: (true, true),
            },
            Support {
                node: nodes[3].clone(),
                value: (true, false),
            },
        ],
    );

    let _ = structure.plot_topology();

    structure.solve(440, 10, 0.054545, 1e-4);
}
