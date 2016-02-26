package twitter4j.examples;

/*
 * Copyright 2007 Yusuke Yamamoto
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import twitter4j.Status;
import twitter4j.TwitterException;
import twitter4j.TwitterObjectFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Example application that load raw JSON forms from statuses/ directory and dump status texts.
 *
 * @author Yusuke Yamamoto - yusuke at mac.com
 */
public final class LoadRawJSONClass {
    /**
     * Usage: java twitter4j.examples.json.LoadRawJSON
     *
     * @param args String[]
     */
    public static List<Status> load(String filename) {
        try {
            File file = new File(filename);
            return readLines(file);
        } catch (Exception e) {
        	return null;
        }
    }

    private static List<Status> readLines(File fileName) throws IOException {
        FileInputStream fis = null;
        InputStreamReader isr = null;
        BufferedReader br = null;
        List<Status> ret = new ArrayList<Status>();
        try {
            fis = new FileInputStream(fileName);
            isr = new InputStreamReader(fis, "UTF-8");
            br = new BufferedReader(isr);
            String nextLine = br.readLine();
            while(nextLine!=null) {
            	try {
					ret.add(TwitterObjectFactory.createStatus(nextLine));
				} catch (TwitterException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
            	nextLine = br.readLine();
            }
         return ret;
            
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
}