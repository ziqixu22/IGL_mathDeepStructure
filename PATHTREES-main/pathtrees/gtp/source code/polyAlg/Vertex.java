/** This file is part of GTP, a program for computing the geodesic distance between phylogenetic trees,
 * and sturmMean, a program for computing the Frechet mean between phylogenetic trees.
    Copyright (C) 2008-2012  Megan Owen, Scott Provan

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>. */

package polyAlg;

/**Defines a vertex,with the appropriate objects */

public class Vertex {
	public double label, weight, residual; // -1 unlabeled, otherwise max flow to that vertex
	public int pred; //-1 = unscanned, otherwise predecessor
	
	public Vertex(double weight) {this.weight=weight*weight;};
		
	}