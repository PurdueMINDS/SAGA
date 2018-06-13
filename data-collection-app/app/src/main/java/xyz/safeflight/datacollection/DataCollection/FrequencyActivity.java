package xyz.safeflight.datacollection.DataCollection;

import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.support.v4.app.TaskStackBuilder;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.app.NotificationCompat;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import xyz.safeflight.datacollection.FileManagement.FileSingleton;
import xyz.safeflight.datacollection.R;
import xyz.safeflight.datacollection.Utils;

public class FrequencyActivity extends AppCompatActivity {
    private final String TAG = "Frequency Activity";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_frequency);


    }

    public void handleStartButton(View v){
        final TextView text = (TextView) findViewById(R.id.editText);
        int period = Integer.parseInt(text.getText().toString());

        // Send user selected period to new activity
        Intent intent = new Intent(this, CollectData.class);
        intent.putExtra("period",period);

        // Create new file name = time
        FileSingleton file = FileSingleton.getInstance();
        file.setFilename(Utils.getTime("HH_mm_ss"));

        Log.e(TAG,"Filename is: " + file.getDirectory() + file.getFilename());

        startActivity(intent);
    }


}
