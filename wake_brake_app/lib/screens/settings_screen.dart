import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/config_provider.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final config = ref.watch(configProvider);
    final notifier = ref.read(configProvider.notifier);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text("Alert Configuration", style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 20),
          Card(
            child: ListTile(
              title: const Text("🔊 Sound Alert"),
              trailing: Switch(
                value: config.soundEnabled,
                onChanged: (val) => notifier.updateSound(val),
              ),
            ),
          ),
          const SizedBox(height: 10),
          Card(
            child: ListTile(
              title: const Text("📳 Vibration Alert"),
              trailing: Switch(
                value: config.vibrationEnabled,
                onChanged: (val) => notifier.updateVibration(val),
              ),
            ),
          ),
          const SizedBox(height: 10),
          Card(
            color: config.isScentInCooldown ? Colors.grey.shade800 : null,
            child: Column(
              children: [
                ListTile(
                  title: const Text("🌫️ Scent Spray"),
                  subtitle: config.isScentInCooldown
                      ? Text("Cooldown active: ${config.scentCooldownRemaining}s remaining",
                             style: const TextStyle(color: Colors.orange))
                      : const Text("Max 30s ON, followed by 90s cooldown"),
                  trailing: Switch(
                    value: config.scentEnabled,
                    onChanged: config.isScentInCooldown 
                        ? null 
                        : (val) => notifier.updateScent(val),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
