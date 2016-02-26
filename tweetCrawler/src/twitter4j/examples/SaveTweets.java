package twitter4j.examples;

import twitter4j.*;

import java.io.*;
import java.util.List;

public class SaveTweets {
	//Storing function
	//inputs:
	//    currentCount: current total count of tweets save for this filebase
	//    Result: query result to be store at this time. 
	public static void storeQueryResult(String targetFile, QueryResult Result) throws Exception {
		FileOutputStream fos = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        try {
        	//open file with append file set to be true
            fos = new FileOutputStream(targetFile,true);
            osw = new OutputStreamWriter(fos, "UTF-8");
            bw = new BufferedWriter(osw);
            for (Status status : Result.getTweets()) {
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
                }
            }
            if (osw != null) {
                try {
                    osw.close();
                } catch (IOException ignore) {
                }
            }
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException ignore) {
                }
            }
        }
	}  
}