import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_background_service/flutter_background_service.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';
import 'screens/main_ui.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // 1. Request notification permission for Android 13+
  await Permission.notification.request();
  
  // 2. Create the notification channel (Required for Android 8+)
  const AndroidNotificationChannel channel = AndroidNotificationChannel(
    'wake_brake_channel', // id
    'Wake&Brake Service', // title
    description: 'Monitoring driver fatigue', // description
    importance: Importance.low, // low importance so it doesn't pop up every time
  );

  final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
      FlutterLocalNotificationsPlugin();

  await flutterLocalNotificationsPlugin
      .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin>()
      ?.createNotificationChannel(channel);

  // 3. Initialize background service
  await initializeService();
  
  runApp(const ProviderScope(child: WakeBrakeApp()));
}

Future<void> initializeService() async {
  final service = FlutterBackgroundService();
  await service.configure(
    androidConfiguration: AndroidConfiguration(
      onStart: onStart,
      autoStart: true,
      isForegroundMode: true,
      notificationChannelId: 'wake_brake_channel',
      initialNotificationTitle: 'Wake&Brake Service',
      initialNotificationContent: 'Monitoring driver fatigue',
      foregroundServiceNotificationId: 888,
    ),
    iosConfiguration: IosConfiguration(
      autoStart: true,
      onForeground: onStart,
    ),
  );
}

@pragma('vm:entry-point')
void onStart(ServiceInstance service) async {
  service.on('stopService').listen((event) {
    service.stopSelf();
  });

  // Background logic could be here if needed for independent polling
}

class WakeBrakeApp extends StatelessWidget {
  const WakeBrakeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wake&Brake',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        brightness: Brightness.dark,
        colorSchemeSeed: Colors.blue,
      ),
      home: const MainUI(),
    );
  }
}
