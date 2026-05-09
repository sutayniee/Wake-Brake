import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/fatigue_provider.dart';

class MonitorScreen extends ConsumerWidget {
  const MonitorScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(fatigueProvider);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text("Live Monitor", style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 20),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("BPM: ${state.bpm}", style: const TextStyle(fontSize: 16)),
                  const SizedBox(height: 10),
                  Text("EAR: ${state.ear}", style: const TextStyle(fontSize: 16)),
                  const SizedBox(height: 10),
                  Text("Eye Height: ${state.eyeHeight} px", style: const TextStyle(fontSize: 16)),
                  const SizedBox(height: 10),
                  Text("FPS: ${state.fps}", style: const TextStyle(fontSize: 16)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text("Alert Logs", style: Theme.of(context).textTheme.titleLarge),
              TextButton(
                onPressed: () {
                  ref.read(fatigueProvider.notifier).clearLogs();
                },
                child: const Text("Clear"),
              )
            ],
          ),
          const SizedBox(height: 10),
          Card(
            child: SizedBox(
              height: 220,
              width: double.infinity,
              child: state.logs.isEmpty
                  ? const Center(child: Text("No alerts yet"))
                  : ListView.builder(
                      padding: const EdgeInsets.all(12),
                      itemCount: state.logs.length,
                      itemBuilder: (context, index) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 6.0),
                          child: Text("• ${state.logs[index]}"),
                        );
                      },
                    ),
            ),
          )
        ],
      ),
    );
  }
}
