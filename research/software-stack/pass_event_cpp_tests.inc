// Insert inside the existing hlo_pass_pipeline_test.cc anonymous namespace.
std::vector<tsl::profiler::TraceMeRecorder::Event> CollectResearchPassEvents(
    const tsl::profiler::TraceMeRecorder::Events& recorded) {
  std::vector<tsl::profiler::TraceMeRecorder::Event> selected;
  for (const tsl::profiler::TraceMeRecorder::ThreadEvents& thread : recorded) {
    for (const tsl::profiler::TraceMeRecorder::Event& event : thread.events) {
      if (event.name.rfind("research_hlo_pass_run#", 0) == 0) {
        selected.push_back(event);
      }
    }
  }
  return selected;
}

TEST_F(HloPassPipelineTest, ResearchLeafEventsDescribeChangesAndNestedPipeline) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  std::vector<std::string> run_log;
  HloPassPipeline pipeline("research_outer");
  pipeline.AddPass<FooToBarModulePass>();
  HloPassPipeline& nested = pipeline.AddPass<HloPassPipeline>("research_inner");
  nested.AddPass<RecordingPass>("noop", "ran", &run_log);

  ASSERT_TRUE(tsl::profiler::TraceMeRecorder::Start(3));
  absl::StatusOr<bool> result = pipeline.Run(module.get());
  tsl::profiler::TraceMeRecorder::Events recorded =
      tsl::profiler::TraceMeRecorder::Stop();
  ASSERT_OK(result.status());
  EXPECT_TRUE(*result);
  EXPECT_THAT(run_log, ElementsAre("ran"));
  std::vector<tsl::profiler::TraceMeRecorder::Event> events =
      CollectResearchPassEvents(recorded);
  ASSERT_THAT(events, SizeIs(2));
  EXPECT_TRUE(events[0].IsComplete());
  EXPECT_THAT(events[0].name, HasSubstr("pass=foo2bar"));
  EXPECT_THAT(events[0].name, HasSubstr("pipeline=research_outer"));
  EXPECT_THAT(events[0].name, HasSubstr("module=test_module"));
  EXPECT_THAT(events[0].name,
              HasSubstr(absl::StrCat("program_id=", module->unique_id())));
  EXPECT_THAT(events[0].name,
              HasSubstr(absl::StrCat("research_program_id=", module->unique_id())));
  EXPECT_THAT(events[0].name, HasSubstr("status=OK"));
  EXPECT_THAT(events[0].name, HasSubstr("changed=true"));
  EXPECT_TRUE(events[1].IsComplete());
  EXPECT_THAT(events[1].name, HasSubstr("pass=noop"));
  EXPECT_THAT(events[1].name, HasSubstr("research_inner"));
  EXPECT_THAT(events[1].name, HasSubstr("status=OK"));
  EXPECT_THAT(events[1].name, HasSubstr("changed=false"));
}

TEST_F(HloPassPipelineTest, ResearchFilteredLeafDoesNotEmitEvent) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->mutable_config().mutable_debug_options().add_xla_disable_hlo_passes(
      "algsimp");
  std::vector<std::string> run_log;
  HloPassPipeline pipeline("research_filtered");
  pipeline.AddPass<RecordingPass>("algsimp", "filtered", &run_log);
  pipeline.AddPass<RecordingPass>("dce", "ran", &run_log);

  ASSERT_TRUE(tsl::profiler::TraceMeRecorder::Start(3));
  absl::StatusOr<bool> result = pipeline.Run(module.get());
  tsl::profiler::TraceMeRecorder::Events recorded =
      tsl::profiler::TraceMeRecorder::Stop();
  ASSERT_OK(result.status());
  EXPECT_FALSE(*result);
  EXPECT_THAT(run_log, ElementsAre("ran"));
  std::vector<tsl::profiler::TraceMeRecorder::Event> events =
      CollectResearchPassEvents(recorded);
  ASSERT_THAT(events, SizeIs(1));
  EXPECT_TRUE(events[0].IsComplete());
  EXPECT_THAT(events[0].name, HasSubstr("pass=dce"));
  EXPECT_THAT(events[0].name, HasSubstr("changed=false"));
}

TEST_F(HloPassPipelineTest, ResearchErrorEventStopsAndPreservesStatus) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  module->entry_computation()->root_instruction()->SetAndSanitizeName("bar");
  HloPassPipeline pipeline("research_error");
  pipeline.AddPass<BarBlowerUpper>();

  ASSERT_TRUE(tsl::profiler::TraceMeRecorder::Start(3));
  absl::StatusOr<bool> result = pipeline.Run(module.get());
  tsl::profiler::TraceMeRecorder::Events recorded =
      tsl::profiler::TraceMeRecorder::Stop();
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  std::vector<tsl::profiler::TraceMeRecorder::Event> events =
      CollectResearchPassEvents(recorded);
  ASSERT_THAT(events, SizeIs(1));
  EXPECT_TRUE(events[0].IsComplete());
  EXPECT_THAT(events[0].name, HasSubstr("pass=bar-blower-upper"));
  EXPECT_THAT(events[0].name, HasSubstr("status=INTERNAL"));
  EXPECT_THAT(events[0].name, HasSubstr("changed=unknown"));
}

TEST_F(HloPassPipelineTest, ResearchDisabledRecorderPreservesPassResult) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kSimpleModule));
  HloPassPipeline pipeline("research_no_recorder");
  pipeline.AddPass<FooToBarModulePass>();

  ASSERT_TRUE(tsl::profiler::TraceMeRecorder::Start(0));
  absl::StatusOr<bool> result = pipeline.Run(module.get());
  tsl::profiler::TraceMeRecorder::Events recorded =
      tsl::profiler::TraceMeRecorder::Stop();
  ASSERT_OK(result.status());
  EXPECT_TRUE(*result);
  EXPECT_EQ(module->entry_computation()->root_instruction()->name(), "bar");
  EXPECT_TRUE(CollectResearchPassEvents(recorded).empty());
}
