package xyz.safeflight.datacollection.FileManagement;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import xyz.safeflight.datacollection.R;
import xyz.safeflight.datacollection.Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

public class FileActivity extends AppCompatActivity {
    private String TAG = "FileActivity";
	private File flight_file = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_file_data);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG).setAction("Action", null).show();
				shareFile();
			}
        });
        TextView fileView = (TextView) findViewById(R.id.file_view);

        Intent intent = getIntent();
        String filename = intent.getStringExtra("filename");
		flight_file = Utils.getDocumentsDirectory(this, filename);

        plotFile(fileView, filename);
    }

    private void shareFile() {
        if (flight_file == null) return;

//		Uri uri = Uri.fromFile(flight_file);
		Uri uri = FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".fileprovider", flight_file);

		String content_type = "text/csv";

		Log.i(TAG, this.getContentResolver().getType(uri).toString());
		Log.i(TAG, "Clicked on share button: " + uri);
		Log.i(TAG, "Content type: " + content_type);

		// Create send intent, including the file
        Intent sendIntent = new Intent(Intent.ACTION_SEND);
		sendIntent.setFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        sendIntent.setType(content_type);
		sendIntent.setData(uri);
		sendIntent.putExtra(Intent.EXTRA_STREAM, uri);

		// Always use string resources for UI text.
        // This says something like "Share this photo with"
        String title = getResources().getString(R.string.chooser_title);

        // Create intent to show the chooser dialog
        Intent chooser = Intent.createChooser(sendIntent, title);

        // Verify the original intent will resolve to at least one activity
        if (sendIntent.resolveActivity(getPackageManager()) != null) {
            startActivity(chooser);
        }
	}


    private void plotFile (TextView fileView, String filename) {
        File f = Utils.getDocumentsDirectory(this, filename);
        FileInputStream iStr = null;
        try {
            iStr = new FileInputStream (f);
        } catch (FileNotFoundException e) {
            Log.e(TAG,"File does not exist");
            Toast.makeText(getApplicationContext(), "File does not exist !", Toast.LENGTH_LONG).show();
            finish();
        }
        InputStreamReader isr = new InputStreamReader(iStr);
        BufferedReader bufferedReader = new BufferedReader(isr);
        StringBuilder sb = new StringBuilder();
        String line;
        try {
            while ((line = bufferedReader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            fileView.setText(sb);
            bufferedReader.close();
            iStr.close();
        } catch (IOException e) {
            Log.e(TAG,"Impossible operation with this flight_file");
        }

    }
}
