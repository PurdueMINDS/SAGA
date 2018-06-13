package xyz.safeflight.datacollection.FileManagement;

import android.os.Environment;

import java.io.File;

/**
 * Created by migue on 10/13/2016.
 */
public class FileSingleton {
    private String filename;
    private String directory;

    public String getFilename() {return filename;}

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public String getDirectory() {
        return directory + "/";
    }

    public void setDirectory(String directory) {
        this.directory = directory;
    }

    private static final FileSingleton holder = new FileSingleton();

    public static FileSingleton getInstance() {
        return holder;
    }

    public File getExternalDirectory(){
        return Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);

    }
}
