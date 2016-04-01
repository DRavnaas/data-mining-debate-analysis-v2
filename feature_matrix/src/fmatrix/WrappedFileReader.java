package fmatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class WrappedFileReader {
	public static File getFileByNameString(String filePath) {
		File fileOBJ = new File(filePath);
		if (fileOBJ.exists())
			return fileOBJ;
		else
			return null;
	}
	
	//read txt file wrapper, it takes a method as argument to process each line 
	public static int readTxtFile(String filePath,Method handlerMethod,Object handlerObj,boolean hasHeader) throws Exception{
		File freqWdFile = getFileByNameString(filePath);
		if (freqWdFile == null) {
			System.out.println("no config file found: "+filePath);
			return -1;}
		FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null;
        try {
            fis = new FileInputStream(freqWdFile);
            isr = new InputStreamReader(fis, "UTF-8");
            br = new BufferedReader(isr);
            if (hasHeader)
            	br.readLine();
            String nextLine = br.readLine();
            while(nextLine!=null) {
            	Object[] parameters = new Object[1];
                parameters[0] = nextLine;
                //invoke the given method to process current line
            	handlerMethod.invoke(handlerObj, parameters);
            	nextLine = br.readLine();
            }
            return 0;
            
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException ignore) {
                }
            }
            if (isr != null) {
                try {
                    isr.close();
                } catch (IOException ignore) {
                }
            }
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException ignore) {
                }
            }
        }
	}
	
	public static ArrayList<String[]> readCsvFile(String inputCSV) throws Exception {
		File inputFile = getFileByNameString(inputCSV);
	    if (inputFile == null) {
	    	System.out.println("no input csv file found");
	    	return null;
	    }
	    ArrayList<String[]> ret = new ArrayList<String[]>();
	    CSVReader reader = null;
	    try {
	    	reader = new CSVReader(new FileReader(inputFile));
	    	String [] inputCsvHeader;
	    	inputCsvHeader = reader.readNext();
	    	ret.add(inputCsvHeader);
	    	String[] nextLine;
		    while ((nextLine = reader.readNext()) != null) {
		    	ret.add(nextLine);
		    	
		    }
		    reader.close();
		    reader = null;
	        
	        return ret;
	    }
	    finally {
	    	if (reader != null) {
	            try {
	           	 reader.close();
	            } catch (IOException ignore) {
	            }
	         }
	    } 
	}
	
	public static int writeCsvFile(String outputCSV, ArrayList<String[]> outputData) throws Exception {
    File outputFile = new File(outputCSV);
    CSVWriter writer = null;
    try {
    	writer = new CSVWriter(new FileWriter(outputFile));
    	for (int i =0;i<outputData.size();i++) {
    		writer.writeNext(outputData.get(i));
    	}
	    
	    writer.flush();
      	writer.close();
      	writer = null;
        return 0;
    }
    finally {
    	if (writer != null) {
            try {
           	 writer.flush();
           	 writer.close();
            } catch (IOException ignore) {
            }
         }
        
    }
	}
}