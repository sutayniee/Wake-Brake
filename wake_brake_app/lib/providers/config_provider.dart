import 'dart:async';
import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

class ConfigState {
  final bool soundEnabled;
  final bool vibrationEnabled;
  final bool scentEnabled;
  final bool isScentInCooldown;
  final int scentCooldownRemaining;

  ConfigState({
    this.soundEnabled = true,
    this.vibrationEnabled = true,
    this.scentEnabled = true,
    this.isScentInCooldown = false,
    this.scentCooldownRemaining = 0,
  });

  ConfigState copyWith({
    bool? soundEnabled,
    bool? vibrationEnabled,
    bool? scentEnabled,
    bool? isScentInCooldown,
    int? scentCooldownRemaining,
  }) {
    return ConfigState(
      soundEnabled: soundEnabled ?? this.soundEnabled,
      vibrationEnabled: vibrationEnabled ?? this.vibrationEnabled,
      scentEnabled: scentEnabled ?? this.scentEnabled,
      isScentInCooldown: isScentInCooldown ?? this.isScentInCooldown,
      scentCooldownRemaining:
          scentCooldownRemaining ?? this.scentCooldownRemaining,
    );
  }
}

class ConfigNotifier extends StateNotifier<ConfigState> {
  Timer? _cooldownTimer;

  ConfigNotifier() : super(ConfigState());

  void updateSound(bool value) {
    state = state.copyWith(soundEnabled: value);
    _syncWithServer();
  }

  void updateVibration(bool value) {
    state = state.copyWith(vibrationEnabled: value);
    _syncWithServer();
  }

  void updateScent(bool value) {
    if (state.isScentInCooldown && value) {
      return;
    }
    state = state.copyWith(scentEnabled: value);
    _syncWithServer();

    if (value) {
      _startScentCycle();
    } else {
      _cooldownTimer?.cancel();
    }
  }

  void _startScentCycle() {
    _cooldownTimer?.cancel();
    _cooldownTimer = Timer(const Duration(seconds: 30), () {
      state = state.copyWith(
        scentEnabled: false,
        isScentInCooldown: true,
        scentCooldownRemaining: 90,
      );
      _syncWithServer();

      _cooldownTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
        if (state.scentCooldownRemaining > 0) {
          state = state.copyWith(
            scentCooldownRemaining: state.scentCooldownRemaining - 1,
          );
        } else {
          state = state.copyWith(isScentInCooldown: false);
          timer.cancel();
        }
      });
    });
  }

  Future<void> _syncWithServer() async {
    try {
      await http.post(
        //change 1231231321
        Uri.parse('http://192.168.1.10:5000/config'),
        headers: {"Content-Type": "application/json"},
        body: json.encode({
          "sound": state.soundEnabled,
          "vibration": state.vibrationEnabled,
          "scent": state.scentEnabled,
        }),
      );
    } catch (e) {
      // ignore: avoid_print
      debugPrint("Failed to sync config: $e");
    }
  }
}

final configProvider = StateNotifierProvider<ConfigNotifier, ConfigState>((
  ref,
) {
  return ConfigNotifier();
});
