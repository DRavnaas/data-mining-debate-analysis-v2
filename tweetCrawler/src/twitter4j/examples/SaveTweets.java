package twitter4j.examples;

import twitter4j.*;

import java.io.*;
import java.util.List;

public class SaveTweets {
	//Storing function
	//inputs:
	//    targetFile: 
	//    Result: query result to be store at this time.
	// 	  apendToFile: true if the input tweets should be appended to the input file contents.
	//                 false if hte input tweets should be written to the file, overwriting any existing contents.
	public static void storeQueryResult(String targetFile, QueryResult Result, boolean appendToFile) throws Exception {
		FileOutputStream fos = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        boolean firstWrite = true;
        
        if (!appendToFile)
        {
        	System.out.println("Writing " + Result.getTweets().size() + " tweets to file: " + targetFile);
        }
        else {
        	System.out.println("Appending " + Result.getTweets().size() + " tweets to file: " + targetFile);
        	
        }
        
 
        
        try {
        	//open file with append file set to as specified
            fos = new FileOutputStream(targetFile, appendToFile);
            osw = new OutputStreamWriter(fos, "UTF-8");
            bw = new BufferedWriter(osw);
            for (Status status : Result.getTweets()) {
            	
            	if ((firstWrite) && (!appendToFile))
            	{
            		// Start a new file = write a header, removing any existing contents.
            		bw.write(Result.getQuery());
            		bw.newLine();
            		firstWrite = false;
            	}
            	
            	String rawJSON = TwitterObjectFactory.getRawJSON(status);
            	
            	bw.write(rawJSON);
                bw.newLine();
                bw.flush();
           }    
        }catch (Exception e) {
            e.printStackTrace();
            System.out.println("Failed to store tweets: " + e.getMessage());
            throw e;
        } finally {
            if (bw != null) {
                try {
                    bw.close();
                } catch (IOException ignore) {
                    System.out.println("Ignoring exception when closing writer: " + ignore.getMessage());
                }
            }
            if (osw != null) {
                try {
                    osw.close();
                } catch (IOException ignore) {
                    System.out.println("Ignoring exception when closing writer: " + ignore.getMessage());
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException ignore) {
                    System.out.println("Ignoring exception when closing writer: " + ignore.getMessage());
                }
            }
        }
	}  
}