import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainParser
{

    public static void main(String[] args) throws IOException
    {
        // TODO Auto-generated method stub

        // Pattern String Definitions
        String _empl = "\\s*\r|\\s*\n";
        String _html = "(http://|https://|ftp://|www)[A-Za-z0-9_\\./=@\\+\\-\\;]*";
        String _stop = "[a-z0-9]+[a-zA-Z0-9_]*\\.\\s*"; // '+' for float

        // Pattern Instances Definitions
        Pattern p_empl = Pattern.compile(_empl);
        Pattern p_html = Pattern.compile(_html);
        Pattern p_stop = Pattern.compile(_stop);

        // Loop to read files

        String origin_folder = new String("/gov2-analysis/");
        String output_folder = new String("/gov2-analysis2/");
        File[] folder = new File(origin_folder.toString()).listFiles();

		FileWriter debug =  new FileWriter (new File("/debug"));
		

        for (File subfolder : folder)
        {
            if (subfolder.isDirectory())
            {
                String newfolder = new String(output_folder + subfolder.getName());
                File[] infolder = subfolder.listFiles();

                for (File inf : infolder)
                {
                    String newfile = new String(newfolder + "/" + inf.getName());
					
					debug.write(inf.getName());
					debug.flush();
					File test_f = new File(newfile);
					if(test_f.exists())
					   continue;

                    BufferedReader br = new BufferedReader(new FileReader(inf));
                    StringBuilder sb = new StringBuilder();
                    String line = br.readLine();
                    int lineno = 1;

                    while (line != null)
                    {

                        Matcher m_empl = p_empl.matcher(line);
                        if(!m_empl.find() && line.length() != 0) // non-empty line
                        {
                            if(lineno != 1 && lineno != 2)
                            {
                                // filter html
                                Matcher m_html = p_html.matcher(line);
                                int last_start = 0;
                                String newline = "";
                                while(m_html.find())
                                {
                                    newline += line.substring(last_start, m_html.start());
                                    last_start = m_html.end();
                                }
                                newline += line.substring(last_start, line.length());
                                line = newline;

                                // filter full stop
                                Matcher m_stop = p_stop.matcher(line);
                                last_start = 0;
                                newline = "";
                                while(m_stop.find())
                                {
                                    newline += line.substring(last_start, m_stop.end() - 2) + " ";
                                    last_start = m_stop.end();
                                }
                                newline += line.substring(last_start, line.length());
                                line = newline;

                                line = line.replaceAll("[\\-]", "");  // replace - with ""
                                line = line.replaceAll("[\\:\\&\\_\\/\"]", " "); // replace :,&,_,/," with " "
								line = line.replaceAll("[0-9]","");
                            }
                            sb.append(line.trim());
                            //sb.append(System.lineSeparator());
                            sb.append("\n");
                            lineno++;
                            //System.out.println(lineno+++": " +line);
                        }
                        line = br.readLine();

						while(line != null && line.length() > 600)
						   line = br.readLine();
                    }

                    br.close();
                    //System.out.print(sb.toString());
                    FileWriter Output = new FileWriter (new File(newfile));
                    Output.write(sb.toString());
                    Output.close();
					
					debug.write(" !\n");
					debug.flush();
                }
            }
        }
		debug.close();
    }
}
