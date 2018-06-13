package xyz.safeflight.datacollection;

/**
 * Created by migue on 9/10/2016.
 */
public class Constants {
    // NO GOOD PROGRAMMING PRACTICES VALUE OF CONSTANT ARE VISIBLE FOR ALL CLASSES
    // CHANGE TO LOCAL VALUES IN FUTURE
    static final public String RESULT = "REQUEST_PROCESSED";
    static final public String NO_COMMUNICATION = "STOP_COMM";
    static final public String BACK_TO_WORK = "BACK_TO_WORK";
    static final public String PLOT = "PLOT";
    static final public String PLOT_VALUE = "PLOT_VALUE";
    static final public String RESUME = "RESUME";
    static final public String MESSAGE = "LocationMessage";
    static final public String TURN_ON_LOC = "TurnLocationOn";
    static final public Integer MAX_LOCATIONS = 5;
    static final public Integer ACCELEROMETER = 1;
    static final public Integer BAROMETER = 2;
    static final public Integer GYROSCOPE = 3;
    // Sensors get data sensorAccuracy times faster than GPS
    static final public Integer SENSORACCURACY = 4;

}
