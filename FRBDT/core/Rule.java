/***********************************************************************

	This file is part of KEEL-software, the Data Mining tool for regression, 
	classification, clustering, pattern mining and so on.

	Copyright (C) 2004-2010
	
	F. Herrera (herrera@decsai.ugr.es)
    L. Sánchez (luciano@uniovi.es)
    J. Alcalá-Fdez (jalcala@decsai.ugr.es)
    S. García (sglopez@ujaen.es)
    A. Fernández (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see http://www.gnu.org/licenses/
  
**********************************************************************/

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Rule.java
 *    Copyright (C) 2001 University of Waikato, Hamilton, New Zealand
 */

package keel.Algorithms.Decision_Trees.FRBDT.core;

import java.io.Serializable;

/**
 * Abstract class of generic rule
 *
 * @author Xin Xu (xx5@cs.waikato.ac.nz)
 * @version $Revision: 1.8 $
 */
public abstract class Rule implements Copyable, Serializable {

    /** for serialization */
    private static final long serialVersionUID = 8815687740470471229L;

    /**
     * Get a shallow copy of this rule
     *
     * @return the copy
     */
    public Object copy(){ return this;}

    /**
     * Whether the instance covered by this rule
     *
     * @param datum the instance in question
     * @return the boolean value indicating whether the instance
     *         is covered by this rule
     */
    public abstract boolean covers(Instance datum);

    /**
     * Build this rule
     *
     * @param data the data used to build the rule
     * @exception Exception if rule cannot be built
     */
    public abstract void grow(Instances data) throws Exception;

    /**
     * Whether this rule has antecedents, i.e. whether it is a default rule
     *
     * @return the boolean value indicating whether the rule has antecedents
     */
    public abstract boolean hasAntds();

    /**
     * Get the consequent of this rule, i.e. the predicted class
     *
     * @return the consequent
     */
    public abstract double getConsequent();

    /**
     * The size of the rule.  Could be number of antecedents in the case
     * of conjunctive rule
     *
     * @return the size of the rule
     */
    public abstract double size();
}
