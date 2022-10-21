import numpy as np
import pysmsh

class Reader:

    def read(self, file_name, scalar_fields=None, vector_fields=None, axis_names=("x", "y", "z")):

        data = np.load(file_name)

        # List of (name, coordinate values) for each axis
        axes = [(name, data[name]) for name in axis_names]

        # Number of ghost cells
        num_ghosts = data["num_ghost_cells"][0]

        if not np.all(data["num_ghost_cells"] == num_ghosts):
            raise ValueError("Differing depths of the ghost cell region not supported")

        # Create the mesh
        self.mesh = pysmsh.Mesh.Rectilinear(axes,
                                            num_ghost_cells=num_ghosts,
                                            includes_ghost_coords=True)

        # Inject scalar fields
        if scalar_fields is not None:
            for name, coloc in scalar_fields.items():

                # Create the field
                setattr(self, name, pysmsh.Field.Scalar(self.mesh, coloc))

                field = getattr(self, name)
                field.data = data[name]

        # Inject vector fields
        if vector_fields is not None:
            for name, coloc in vector_fields.items():

                # Create the field
                setattr(self, name, pysmsh.Field.Vector(self.mesh, coloc))

                # Transfer data
                field = getattr(self, name)
                field.data = tuple([data[name + c] for c in axis_names])
