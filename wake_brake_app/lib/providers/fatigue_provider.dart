import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/legacy.dart';
import 'package:http/http.dart' as http;
import '../utils/parsing_utils.dart';
import '../services/api_service.dart';

class FatigueState {
  final String fatigue;
  final String bpm;
  final String ear;
  final String eyeHeight;
  final String fps;
  final List<String> logs;

  FatigueState({
    this.fatigue = "Loading...",
    this.bpm = "0",
    this.ear = "0.00",
    this.eyeHeight = "0",
    this.fps = "0",
    this.logs = const [],
  });

  FatigueState copyWith({
    String? fatigue,
    String? bpm,
    String? ear,
    String? eyeHeight,
    String? fps,
    List<String>? logs,
  }) {
    return FatigueState(
      fatigue: fatigue ?? this.fatigue,
      bpm: bpm ?? this.bpm,
      ear: ear ?? this.ear,
      eyeHeight: eyeHeight ?? this.eyeHeight,
      fps: fps ?? this.fps,
      logs: logs ?? this.logs,
    );
  }
}

class FatigueNotifier extends StateNotifier<FatigueState> {
  Timer? _timer;
  String _lastStatus = "";

  FatigueNotifier() : super(FatigueState()) {
    _startPolling();
  }

  void _startPolling() {
    _timer = Timer.periodic(const Duration(milliseconds: 500), (timer) async {
      try {
        final response = await http.get(
          Uri.parse('http://192.168.1.10:5000/fatigue'),
        );
        if (response.statusCode == 200) {
          final jsonStr = ParsingUtils.cleanWhitespace(response.body);
          final data = json.decode(jsonStr);

          final String currentFatigue = data['level'] ?? 'UNKNOWN';
          final String fatigueLog = data['fatigue_log'] ?? '';

          final double earVal = (data['ear'] ?? 0.0).toDouble();
          final double eyeHeightVal = (data['eye_height'] ?? 0.0).toDouble();
          final double fpsVal = (data['fps'] ?? 0.0).toDouble();
          final double bpmVal = (data['bpm'] ?? 0.0).toDouble();

          List<String> newLogs = List.from(state.logs);

          if (currentFatigue != _lastStatus) {
            final time = _getCurrentTime();
            switch (currentFatigue) {
              case 'WARNING_HAPTIC':
                newLogs.add('[$time] Drowsy Warning Detected');
                newLogs.add('[$time] Haptic Alert Activated');
                break;
              case 'CRITICAL_BUZZER':
                newLogs.add('[$time] Critical Fatigue Detected');
                newLogs.add('[$time] Buzzer + Vibration Activated');
                break;
              case 'SEVERE_SCENT':
                newLogs.add('[$time] Severe Fatigue Detected');
                newLogs.add('[$time] Scent Alert Activated');
                break;
              case 'SAFE':
                newLogs.add('[$time] Driver Alerted and Safe');
                break;
            }
            if (fatigueLog.isNotEmpty) {
              newLogs.add('[$time] $fatigueLog');
            }
            _lastStatus = currentFatigue;
          }

          state = state.copyWith(
            fatigue: currentFatigue,
            ear: earVal.toStringAsFixed(2),
            eyeHeight: eyeHeightVal.toStringAsFixed(0),
            fps: fpsVal.toStringAsFixed(1),
            bpm: bpmVal.toStringAsFixed(0),
            logs: newLogs,
          );
        } else {
          state = state.copyWith(fatigue: "ERROR");
        }
      } catch (e) {
        state = state.copyWith(fatigue: "ERROR");
      }
    });
  }

  String _getCurrentTime() {
    final now = DateTime.now();
    return "${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:${now.second.toString().padLeft(2, '0')}";
  }

  void clearLogs() {
    state = state.copyWith(logs: []);
  }

  void panicReset() {
    ApiService.panicReset();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }
}

final fatigueProvider = StateNotifierProvider<FatigueNotifier, FatigueState>((
  ref,
) {
  return FatigueNotifier();
});
