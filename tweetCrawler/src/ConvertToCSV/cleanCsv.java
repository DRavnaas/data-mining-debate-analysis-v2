package ConvertToCSV;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;

public class cleanCsv
{

    // Cleans a csv file of line breaks in the middle of tweet text
    // (these cause ingestion issues)
    public static void main(String[] args) throws Exception
    {
        if (args.length < 2)
        {
            System.out.println("CleanCsv <input csv> <output csv> <id offset #>");
            return;
        }

        int idOffset = 0;
        if (args.length > 2)
        {
            idOffset = Integer.parseInt(args[2]);
        }

        File src = new File(args[0]);
        File dest = new File(args[1]);
        String absolutePath = src.getAbsolutePath();
        System.out.println("Reading file at " + absolutePath);

        CSVReader reader = null;
        CSVWriter writer = null;

        try
        {
            String[] headerFields = null;

            // First line is a header
            reader = new CSVReader(new FileReader(src.getAbsolutePath()));
            writer = new CSVWriter(new FileWriter(dest, false));

            headerFields = reader.readNext();

            if (headerFields == null || headerFields.length <= 0)
            {
                throw new Exception("Could not read header line from csv");
            } else
            {
                System.out.println("Ingesting " + args[0]);
            }

            if (headerFields[0].startsWith("﻿"))
            {
                // I think these are byte markers for UTF-8? Strip them off.
                headerFields[0] = headerFields[0].substring(3);
            }

            // Add a id field to the csv.
            String[] newHeader = new String[headerFields.length + 1];
            newHeader[0] = "id";
            int textIndex = -1;
            boolean idFound = false;
            int expectedFieldLength = headerFields.length;
            for (int i = 0; i < headerFields.length; i++)
            {
                newHeader[i + 1] = headerFields[i];

                if (headerFields[i].equalsIgnoreCase("text"))
                {
                    textIndex = i;

                }

                if (headerFields[i].equalsIgnoreCase("id"))
                {
                    idFound = true;
                }
            }

            // if we need to insert an id field, use the new header we built.
            if (!idFound)
            {
                textIndex = textIndex + 1;
                // expectedFieldLength = expectedFieldLength + 1;

                headerFields = newHeader;
            }

            if (textIndex == -1)
            {
                throw new Exception("No text column found!");
            }

            writer.writeNext(headerFields);

            int id = idOffset;
            String[] fields = null;
            while ((fields = reader.readNext()) != null)
            {
                id++;

                if ((expectedFieldLength != fields.length) || fields[0].isEmpty())
                {
                    System.err.println("Header fields must match # of fields; first field can't be empty!");
                    System.err.println(fields[0]);

                    continue;
                }

                // the header got inserted a couple times in the
                // json to csv code.
                if (fields[0].equalsIgnoreCase("tweet_id"))
                {
                    continue;
                }

                // A couple fields have the string "null" which R doesn't like.
                if (fields[5].equalsIgnoreCase("null"))
                {
                    fields[5] = "none";
                }
                if (fields[6].equalsIgnoreCase("null"))
                {
                    fields[6] = "none";
                }

                switch (fields[1])
                {
                case "Trump":
                    fields[1] = "Donald Trump";
                    break;
                case "Rubio":
                    fields[1] = "Marco Rubio";
                    break;
                case "Cruz":
                    fields[1] = "Ted Cruz";
                    break;
                case "Kasich":
                    fields[1] = "John Kasich";
                    break;
                default:
                    fields[1] = "none";
                    break;
                }

                if (!idFound)
                {
                    String[] newFields = new String[fields.length + 1];
                    newFields[0] = Integer.toString(id);

                    for (int i = 0; i < fields.length; i++)
                    {
                        newFields[i + 1] = fields[i];
                    }

                    fields = newFields;
                }

                // A couple cleanups on the text column so we can ingest as a csv in r or excel
                String text = new String(fields[textIndex]);

                // A couple utf-8 representations of the "this tweet was truncated" character that
                // might sneak in

                if (text.length() > 140)
                {
                    int c = (int) text.charAt(139);
                    int c2 = (int) text.charAt(140);

                    if ((c == 0xFFFD) && (c2 == 0xFFFD))
                    {
                        fields[textIndex] = fields[textIndex].substring(0, 138);
                    } else if (c == 0xFFFD)
                    {
                        fields[textIndex] = fields[textIndex].substring(0, 139);
                    }
                }
                fields[textIndex].replace("<U+FFFD>", " ");
                fields[textIndex].replace("\uFFFD", " ");
                fields[textIndex].replace("\n", " ");
                fields[textIndex].replace("\r", " ");

                writer.writeNext(fields);
            }

            System.out.println("Done with ingestion of " + args[0] + ", output written to " + args[1]);

        }
        catch (Exception e)
        {
            e.printStackTrace();
            throw e;
        }
        finally
        {
            if (reader != null)
            {
                reader.close();
            }

            if (writer != null)
            {
                writer.close();
            }
        }

    }

}
