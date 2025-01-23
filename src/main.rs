use ndarray::{arr1, arr2, array, Array1, Array2};
use plotters::prelude::*;

#[derive(Debug)]
struct Node {
    coord: (f64,f64),
    dofs: (usize,usize)
}

#[derive(Debug)]
struct LagrangianBarElement<'a> {
    nodes: (&'a Node, &'a Node),
    e_module: f64,
    area: f64,
}

impl LagrangianBarElement<'_> {

    fn length(&self) -> f64 {
        ((self.nodes.1.coord.0-self.nodes.0.coord.0).powf(2.0) + (self.nodes.1.coord.1-self.nodes.0.coord.1).powf(2.0)).sqrt()
    }

    fn dof_vector(&self) -> Array1<usize>{
        arr1(&[self.nodes.0.dofs.0,self.nodes.0.dofs.1,self.nodes.1.dofs.0,self.nodes.1.dofs.1])
    }

    fn dofs(&self) -> [usize; 4]{
        [self.nodes.0.dofs.0,self.nodes.0.dofs.1,self.nodes.1.dofs.0,self.nodes.1.dofs.1]
    }

    fn dof_mask(&self) -> [[usize; 4]; 2] {
        [self.dofs(),self.dofs()]
    }

    fn deformation_interpolation_matrix(&self, x_l: f64) -> Array2<f64> {
        let n1: f64 = x_l/self.length();
        let n0: f64 = 1.0-&n1;

        array![[n0,0.0,n1,0.0],
               [0.0,n0,0.0,n1]]
    }

    fn strain_interpolation_matrix(&self) -> Array2<f64> {
        let k: f64 = 1.0/self.length();

        arr2(&[[-1.0, 0.0, 1.0, 0.0]]) * k
    }

    fn transformation_matrix(&self) ->  Array2<f64> {
        let l: f64 = self.length();
        let nx: f64 = (self.nodes.1.coord.0-self.nodes.0.coord.0)/l;
        let ny: f64 = (self.nodes.1.coord.1-self.nodes.0.coord.1)/l;

        arr2(&[
            [ nx,  ny, 0.0, 0.0],
            [-ny,  nx, 0.0, 0.0],
            [0.0, 0.0,  nx,  ny],
            [0.0, 0.0, -nx,  ny],
        ])
    }

    fn axial_force(&self, ve: &Vec<f64>) -> Array2<f64> {
        let ve: Array2<f64> = Array2::from_shape_vec((4,1), ve.clone()).unwrap();
        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let (gx,_,g): (Array2<f64>,Array2<f64>,Array2<f64>) = self.g_parts();
        let local_ve: Array2<f64> = transformation_matrix.dot(&ve);
        let local_strain_linear: Array2<f64> =  gx.dot(&local_ve);
        let local_strain_non_linear: Array2<f64> = 0.5*(&local_ve.t().dot(&g)).dot(&local_ve);

        self.area*self.e_module*(local_strain_linear+local_strain_non_linear)
    }

    fn g_parts(&self) -> (Array2<f64>,Array2<f64>,Array2<f64>) {
        let gx = self.strain_interpolation_matrix();
        let gy = -gx.clone();
        let g= &gx.t().dot(&gx) + &gy.t().dot(&gy);
        (gx,gy,g)
    }

    fn element_tangent_stiffness(&self, ve: &Vec<f64>) -> Array2<f64> {
        // ve should be of length 4

        let l: f64 = self.length();
        let ea: f64 = self.area*self.e_module;
        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let (gx,_,g): (Array2<f64>, Array2<f64>,Array2<f64>) = self.g_parts();
        
        let axial_force: Array2<f64> = self.axial_force(&ve);

        let ve: Array2<f64> = Array2::from_shape_vec((4,1), ve.clone()).unwrap();
        let local_ve: &Array2<f64> = &transformation_matrix.dot(&ve);
        
        let k_sigma: Array2<f64> = l*axial_force*&g;

        let bl: Array2<f64> = gx+local_ve.t().dot(&g);
        let k_0: Array2<f64> = l*ea*bl.dot(&bl.t());

        let kl: Array2<f64> = k_sigma+k_0;

        kl*&transformation_matrix.t().dot(&transformation_matrix)    
    }

    fn element_tangent_stiffness_as_dof_index(&self, ve: &Vec<f64> ) -> Vec<((usize, usize), f64)> {
        let k: Array2<f64> = self.element_tangent_stiffness(&ve);
        // let mut global_dof_indicies: [(usize,usize);16] = [(0,0);16];

        let dofs: [usize; 4] = self.dofs();

        // let mut counter: usize = 0;


        // for (col, row) in iproduct!(&dofs, &dofs) {
        //     global_dof_indicies[counter] = (*col,*row);
        //     counter += 1;
        // }

        // (global_dof_indicies,Array1::from_iter(k.iter().cloned()))

        k.indexed_iter().
        map(|((row, col), val)| ((dofs[row], dofs[col]), *val))
        .collect()

    }


    fn element_force_vector(&self, ve: &Vec<f64>) -> Array2<f64> {
        let axial_force: Array2<f64> = self.axial_force(&ve);
        let ve: Array2<f64> = Array2::from_shape_vec((4,1), ve.clone()).unwrap();

        let transformation_matrix: Array2<f64> = self.transformation_matrix();
        let l = self.length();
        let (gx,_,g): (Array2<f64>,_,Array2<f64>) = self.g_parts();
        let local_ve: &Array2<f64> = &transformation_matrix.dot(&ve);

        let element_force_vector: Array2<f64> = l*&axial_force*(&gx.t()+g.dot(local_ve));

        let global_force_vector: Array2<f64> = transformation_matrix.dot(&element_force_vector);

        global_force_vector
        
    }

    fn draw_topology_data(&self) -> [(f64, f64); 2] {
        [(self.nodes.0.coord.0,self.nodes.0.coord.1),(self.nodes.1.coord.0,self.nodes.1.coord.1)]
    }

    fn draw_topology(&self) -> LineSeries<BitMapBackend<'_>, (f64, f64)> {
        LineSeries::new(
            self.draw_topology_data().map(|point| point),
            BLACK.filled()
        ).point_size(4)
    }
}

#[derive(Debug)]
struct Load<'a> {
    node: &'a Node,
    value: (f64,f64),
}

#[derive(Debug)]
struct Support<'a> {
    node: &'a Node,
    value: (bool, bool)
}

#[derive(Debug)]
enum BoundaryCondition<'a> {
    Load(Load<'a>),
    Support(Support<'a>),
}


#[derive(Debug)]
struct Structure<'a> {
    elements: Vec<LagrangianBarElement<'a>>,
    loads: Vec<Load<'a>>,
    supports: Vec<Support<'a>>,
    nno: usize,
}

impl Structure<'_> {
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
            chart
            .draw_series(
                element.draw_topology()
            ).unwrap();
        }
    
        root.present()?;
    
        Ok(())
    }



    fn system_tangent_stiffness_matrix(&self, gve: &Array1<f64>) -> Array2<f64> {
        let mut matrix: Array2<f64> = Array2::zeros((self.nno*2,self.nno*2));

        for element in &self.elements {
            
            let ve: Vec<f64> = element.dofs().iter().map(|dof: &usize| gve[*dof]).collect();

            element.element_tangent_stiffness_as_dof_index(&ve)
            .iter()
            .for_each(|((row, col),val)| matrix[[*row,*col]] += val);
        }
        matrix
    }
}

fn main() {
    const A_DIM: f64 = 3.0;
    const HEIGHT: f64 = 0.2;
    let l0: f64 = ((2.0*A_DIM).powf(2.0)+(HEIGHT).powf(2.0)).sqrt();

    const E: f64 = 2.0;
    const AREA: f64 = 0.25 ;

    let e_norm: f64 = (2.0*A_DIM-l0)/l0;
    let pn: f64 = E*AREA*(e_norm);

    let nodes:[Node; 3] = [
        Node {coord: (-A_DIM,0.0), dofs: (0,1)},
        Node {coord: (0.0,HEIGHT), dofs: (2,3)},
        Node {coord: (A_DIM,0.0), dofs: (4,5)},
        ];

    let nno = nodes.len();

    let structure = Structure {
        elements: vec![
            LagrangianBarElement {nodes: (&nodes[0],&nodes[1]), e_module: E, area: AREA},
            LagrangianBarElement {nodes: (&nodes[1],&nodes[2]), e_module: E, area: AREA},
            ],
        loads: vec![
            Load {node: &nodes[1], value: (0.0,-0.01*pn)},
            ],
        supports: vec![
            Support {node: &nodes[0], value: (true, true)},
            Support {node: &nodes[2], value: (true, true)},
        ],
        nno: nno
    };

    let test_vec0: Vec<f64> = vec![0.0,0.0,0.0,1.0];
    let test_vec1: Vec<f64> = vec![0.0,1.0,0.0,0.0];

    println!("{}\n",structure.elements[0].element_force_vector(&test_vec0));
    println!("{}\n",structure.elements[1].element_force_vector(&test_vec1));

    println!("{}\n",structure.elements[0].element_tangent_stiffness(&test_vec0));
    println!("{}\n",structure.elements[1].element_tangent_stiffness(&test_vec1));

    println!("{:?}\n",structure.elements[0].element_tangent_stiffness_as_dof_index(&test_vec0));
    println!("{:?}\n",structure.elements[1].element_tangent_stiffness_as_dof_index(&test_vec1));

    println!("{}\n",structure.elements[0].dof_vector());
    println!("{}\n",structure.elements[1].dof_vector());

    let test_vec99: Vec<f64> = vec![0.0,0.0,0.0,1.0,0.0,0.0];
    let gve = Array1::from_vec(test_vec99);

    println!("{:?}\n", structure.system_tangent_stiffness_matrix(&gve));

    let _ = structure.plot_topology();
}
