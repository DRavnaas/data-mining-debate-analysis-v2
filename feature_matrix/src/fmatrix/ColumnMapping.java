package fmatrix;
class ColumnMapping {
	public String srcCol;
	public int srcColIdx;
	public String tgtCol;
	boolean isTweetTextField;
	ColumnMapping(String srcCol, String tgtCol, boolean isTweetTextField){
		this.srcColIdx =-1;
		this.srcCol = srcCol;
		this.tgtCol = tgtCol;
		this.isTweetTextField = isTweetTextField;
	}
}