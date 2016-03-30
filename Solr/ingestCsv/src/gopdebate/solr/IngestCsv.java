package gopdebate.solr;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collection;

import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.impl.HttpSolrClient;
import org.apache.solr.client.solrj.response.UpdateResponse;
import org.apache.solr.common.SolrInputDocument;

import com.opencsv.CSVReader;

public class IngestCsv {

    // Input csv must have an id column.
    // All numerics must be formatted to not include commas.
    // First row must be a header row.

    // ToDo: get this from config instead?
    public static String solrUrlBase = "http://192.168.252.98:7574/solr/"; // "http://localhost:8983/solr/";

    public static String collectionName = "gopdebate";
    public static String solrIngestBase = solrUrlBase + collectionName;

    public static void main(String[] args) throws Exception {

        if (args.length < 1) {
            System.err
                    .println("Error, input parameters = file.csv [http://yourhost:8983/solr/yourcollection]");
        }

        if (args.length > 1) {
            solrIngestBase = args[1];
        }

        System.out.println("Pushing csv file " + args[0] + " to solr collection at "
                + solrIngestBase);

        SolrClient solr = new HttpSolrClient(solrIngestBase);

        Collection<SolrInputDocument> docsInChunk = new ArrayList<SolrInputDocument>();

        BufferedReader br = null;
        CSVReader reader = null;

        File f = new File(args[0]);
        try {
            Integer chunkSize = 10000;
            Long numDocs = 0L;
            Long docsSkipped = 0L;

            String[] headerFields = null;

            // First line is a header
            reader = new CSVReader(new FileReader(f.getAbsolutePath()));
            headerFields = reader.readNext();

            if (headerFields == null || headerFields.length <= 0) {
                throw new Exception("Could not read header line from csv");
            } else {
                System.out.println("Ingesting " + args[0]);
            }

            String[] fields = null;
            while ((fields = reader.readNext()) != null) {

                if ((headerFields.length != fields.length) || fields[0].isEmpty()) {
                    System.err.println(fields[0]);
                    // throw new Exception(
                    // "Could not parse input line for given # headers. Line starts with "
                    // + fields[0]);
                    docsSkipped++;
                    continue;
                }
                if (headerFields.length == fields.length) {

                    SolrInputDocument doc = new SolrInputDocument();
                    for (int i = 0; i < headerFields.length; i++) {

                        // Add this field to our current doc to ingest
                        doc.addField(headerFields[i], fields[i]);
                    }

                    docsInChunk.add(doc);
                    numDocs++;

                    if (docsInChunk.size() % chunkSize == 0) {
                        // server.commit();
                        UpdateResponse response = solr.add(docsInChunk);
                        // System.out.println("Update status = " +
                        // response.getStatus() + " for "
                        // + docsInChunk.size() + " docs");
                        if (response.getStatus() != 0) {
                            throw new Exception("Could not add docs - response status = "
                                    + response.getStatus());
                        }
                        if (numDocs % 10000 == 0) {
                            System.out.println("..# docs processed = " + numDocs);

                        }
                        docsInChunk.clear();
                    }
                }
            }

            if (docsInChunk.size() > 0) {
                // One last batch to commit
                UpdateResponse response = solr.add(docsInChunk);
                // System.out.println("Update status = " + response.getStatus()
                // + " for "
                // + docsInChunk.size() + " docs");

                if (response.getStatus() != 0) {
                    throw new Exception("Could not add docs - response status = "
                            + response.getStatus());
                }
                docsInChunk.clear();
            }

            System.out.println("Done with ingestion of " + args[0] + ", # docs ingested = "
                    + numDocs + ", # docs skipped = " + docsSkipped);

        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        } finally {
            if (reader != null) {
                reader.close();
            }
            if (br != null) {
                br.close();
            }

            solr.commit();
            solr.close();
        }
    }
}
