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

package keel.Algorithms.Decision_Trees.FRBDT.core;

import java.util.Enumeration;

/**
 * This class implements a single rule that predicts specified class.  
 *
 * A rule consists of antecedents "AND"ed together and the consequent 
 * (class value) for the classification.  
 * In this class, the Information Gain (p*[log(p/t) - log(P/T)]) is used to
 * select an antecedent and Reduced Error Prunning (REP) with the metric
 * of accuracy rate p/(p+n) or (TP+TN)/(P+N) is used to prune the rule. 
 */    
public class RipperRule extends Rule{
  
  /** for serialization */
  static final long serialVersionUID = -2410020717305262952L;
	
  /** The internal representation of the class label to be predicted */
  public double m_Consequent = -1;	
		
  /** The vector of antecedents of this rule*/
  public FastVector m_Antds = null;

  public double m_alpha=0.02;

  public int m_maxAttUsed=5;

  public double m_shreshold=0.6; 

  /** The minimal number of instance weights within a split*/
  double m_MinNo = 2.0;

  /** Whether in a debug mode */
  protected boolean m_Debug = false;

  /** The class distribution of the training data*/
  double[] aprioriDistribution;
  
  /** Constructor */
  public RipperRule(){    
    m_Antds = new FastVector();
  }
  
  /** Constructor
     * @param aprioriClassDistribution  apriori class distribution to be set.*/
  public RipperRule(double [] aprioriClassDistribution){    
    m_Antds = new FastVector();	
    this.aprioriDistribution = aprioriClassDistribution.clone();
  }
	
  /**
   * Sets the internal representation of the class label to be predicted
   * 
   * @param cl the internal representation of the class label to be predicted
   */
  public void setConsequent(double cl) {
    m_Consequent = cl; 
  }
  
  /**
   * Gets the internal representation of the class label to be predicted
   * 
   * @return the internal representation of the class label to be predicted
   */
  public double getConsequent() { 
    return m_Consequent; 
  }

  public void setMaxAttUsed(int maxl){
	m_maxAttUsed=maxl;
  }

  public int getMaxAttUsed(){ return m_maxAttUsed;}

  public void setAlpha(double alpha){
	m_alpha=alpha;
  }

  public double getAlpha(){return m_alpha;}

  public void setShreshlod(double shreshold){
	m_shreshold=shreshold;
  }

  public double getShreshold(){return m_shreshold;}
	
  /**
   * Get a shallow copy of this rule
   *
   * @return the copy
   */
  public Object copy(){
    RipperRule copy = new RipperRule();
    copy.setConsequent(getConsequent());
    copy.m_Antds = (FastVector)this.m_Antds.copyElements();
    copy.aprioriDistribution = this.aprioriDistribution.clone();
    return copy;
  }
	     
	
  /**
   * Whether this rule has antecedents, i.e. whether it is a default rule
   * 
   * @return the boolean value indicating whether the rule has antecedents
   */
  public boolean hasAntds(){
    if (m_Antds == null)
	return false;
    else
	return (m_Antds.size() > 0);
  }      
	
  /** 
   * the number of antecedents of the rule
   *
   * @return the size of this rule
   */
  public double size(){ return (double)m_Antds.size(); }		

	
  /**
   * Private function to compute default number of accurate instances
   * in the specified data for the consequent of the rule
   * 
   * @param is an instance
   * @return the average membership function of the instance in this rule
   */
  private double computeAverageMembershipDegree(Instance is, FastVector antds){ 
    double aveMemDgre=0;
    for(int i=0; i<antds.size(); i++){
		Antd antdi = (Antd)antds.elementAt(i);
		aveMemDgre+=antdi.getMembershipDgree(is);
    }
    return aveMemDgre/antds.size();
  }

  public double computeAverageMembershipDegree(Instance is){ 
    double aveMemDgre=0;
    for(int i=0; i<m_Antds.size(); i++){
		Antd antdi = (Antd)m_Antds.elementAt(i);
		aveMemDgre+=antdi.getMembershipDgree(is);
    }
    return aveMemDgre/m_Antds.size();
  }

  private double computeFuzzyConfidence (Instances data, FastVector antds) throws Exception{
	if(m_Consequent == -1)
	throw new Exception(" Consequent not set yet.");

	Instances Data = data;
	double fuzzyConCovered=0;
	double fuzzyAll=0;
	for (int i=0; i<Data.numInstances();i++){
		Instance ins=Data.instance(i);
		fuzzyAll+=computeAverageMembershipDegree(ins,antds);
		if ((int)ins.classValue()==(int)m_Consequent){
			fuzzyConCovered+=this.computeAverageMembershipDegree(ins,antds);
		}
	}
	return fuzzyConCovered/fuzzyAll;
  }

  private FastVector computeAntdInitial(Instances data, AttributeWeka att){
    FastVector antesIni = new FastVector();

    int numClass =0;
    for (int i=0;i<aprioriDistribution.length;i++){
      if (aprioriDistribution[i]!=0){
        numClass++;
      }
    }
    int [] indexClassNotZero=new int[numClass];
    double [] meanValues = new double [numClass];
    int w=0;
    for (int c=0;c<data.numClasses();c++){
      if (aprioriDistribution[c]!=0){
        indexClassNotZero[w]=c;
        double [] attClassValues = new double [(int)aprioriDistribution[c]];
        int j=0;
        for (int i =0; i<data.numInstances();i++){
          Instance ins = data.instance(i);
          if ((int)ins.classValue()==c){
            attClassValues[j]=ins.value(att);
            j+=1;
          }
        }
        meanValues[w]=Utils.mean(attClassValues);
        w++;
    }
  }
  if (meanValues.length==1){
    double [] keyvalues={Double.NEGATIVE_INFINITY,Double.NEGATIVE_INFINITY,Double.POSITIVE_INFINITY,Double.POSITIVE_INFINITY};
    Antd oneAntd=new Antd(keyvalues, att, indexClassNotZero[0]);
    antesIni.addElement(oneAntd);
  }else{
    int [] sortIndex=Utils.sort(meanValues);
    for (int i=0; i<numClass;i++){
      if (i==0){
        double [] keyvalues={Double.NEGATIVE_INFINITY,Double.NEGATIVE_INFINITY,meanValues[sortIndex[i]],meanValues[sortIndex[i+1]]};
        Antd oneAntd=new Antd(keyvalues, att, indexClassNotZero[sortIndex[i]]);
        antesIni.addElement(oneAntd);
      }else if(i==numClass-1){
        double [] keyvalues={meanValues[sortIndex[i-1]],meanValues[sortIndex[i]],Double.POSITIVE_INFINITY,Double.POSITIVE_INFINITY};
        Antd oneAntd=new Antd(keyvalues, att, indexClassNotZero[sortIndex[i]]);
        antesIni.addElement(oneAntd);
      }else{
        double [] keyvalues={meanValues[sortIndex[i-1]],meanValues[sortIndex[i]],meanValues[sortIndex[i]],meanValues[sortIndex[i+1]]};
        Antd oneAntd=new Antd(keyvalues, att, indexClassNotZero[sortIndex[i]]);
        antesIni.addElement(oneAntd);
      }
    }
  }
  return antesIni;
  }
	
  /**
   * Build one rule using the growing data
   *
   * @param data the growing data used to build the rule
   * @throws Exception if the consequent is not set yet
   */    
    public void grow(Instances data) throws Exception {
    if(m_Consequent == -1)
	throw new Exception(" Consequent not set yet.");
	    
    Instances growData = data;	         
    double sumOfWeights = growData.sumOfWeights();
    if(!Utils.gr(sumOfWeights, 0.0))
	return;
	
    /* Keep the record of which attributes have already been used*/    
    boolean[] used=new boolean [growData.numAttributes()];
    for (int k=0; k<used.length; k++)
	used[k]=false;
    int numUnused=m_maxAttUsed;//用到的属性数量

	double maxFConfFirst=0;
	
	FastVector AntdsFir=new FastVector();
	while (Utils.gr(growData.numInstances(), 0.0)){ 	    
	double maxFConfSecond=0;
	/* Build a list of antecedents */
	Enumeration enumAttr=growData.enumerateAttributes();	      
	
	FastVector antdsSed =new FastVector();
	/* Find the max fuzzy confidence of each attribute*/
	while (enumAttr.hasMoreElements()){
	  AttributeWeka att= (AttributeWeka)(enumAttr.nextElement());
	  
	  if(m_Debug)
	    System.err.println("\nOne condition: size = " 
			       + growData.sumOfWeights());
		    
	  if(!used[att.index()]){//the attribute's index 没有用到.
	    /* Compute the best information gain for each attribute,
	       it's stored in the antecedent formed by this attribute.
	       This procedure returns the data covered by the antecedent*/
		Antd antd=null;
		FastVector antdsIni=computeAntdInitial(growData, att);
		for (int i=0;i<antdsIni.size();i++){
			Antd antdIni = (Antd) antdsIni.elementAt(i);
			if (antdIni.getClassIndex()==m_Consequent){
				antd=antdIni;
        break;
			}
		}
		FastVector oneAntds= (FastVector)AntdsFir.copy();
		oneAntds.addElement(antd);

		double fuzzyConf=computeFuzzyConfidence(growData,oneAntds);  
	    if(fuzzyConf>=maxFConfSecond){         
			antdsSed=oneAntds;  
			maxFConfSecond=fuzzyConf;
	    }		    
	  }
	}

	if(antdsSed.size() == 0) break; // Cannot find antds
	if (maxFConfFirst-maxFConfSecond>=m_alpha){
		m_Antds=AntdsFir;
		break;
	}
	else if(numUnused==1){
		m_Antds=antdsSed;
		break;
	}else{
		maxFConfFirst=maxFConfSecond;
		Antd lastAntd=(Antd) antdsSed.lastElement();
		used[lastAntd.getAttr().index()]=true;
		numUnused--;
		AntdsFir=antdsSed;

	}
    }
  }

  /**
   * Prints this rule
   *
   * @param classAttr the class attribute in the data
   * @return a textual description of this rule
   */
  public String toString(AttributeWeka classAttr) {
    StringBuffer text =  new StringBuffer();
    if(m_Antds.size() > 0){
	for(int j=0; j< (m_Antds.size()-1); j++)
	  text.append("(" + ((Antd)(m_Antds.elementAt(j))).toString()+ ") and ");
	text.append("("+((Antd)(m_Antds.lastElement())).toString() + ")");
    }
    text.append(" => " + classAttr.name() +
		  "=" + classAttr.value((int)m_Consequent));
	    
    return text.toString();
  }

    /**
     * String "1.0"
     * @return "1.0"
     */
    public String getRevision() {
    return "1.0";
  }

  public boolean covers(Instance datum){ 
    if (computeAverageMembershipDegree(datum,m_Antds) <m_shreshold){
	return false;
    }else{
	return true;
    }
  }  
}