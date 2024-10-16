#include <cstdio>
#include <fluidsynth.h>
#include <memory>
 
using namespace std;

int limit = 128;
 
template<typename _T>
using managed_ptr = unique_ptr<_T, void(*)(_T*)>;
 
int main() 
{
    auto settings = managed_ptr<fluid_settings_t>{ 
            new_fluid_settings(), delete_fluid_settings };
// 출력 파일의 이름입니다.
    fluid_settings_setstr(settings.get(), 
            "audio.file.name", "output.wav");
    fluid_settings_setstr(settings.get(), 
            "player.timing-source", "sample");
    fluid_settings_setint(settings.get(), 
            "synth.lock-memory", 0);
 
    auto synth = managed_ptr<fluid_synth_t>{ 
            new_fluid_synth(settings.get()), delete_fluid_synth };
// soundfont 파일을 로드합니다.
    fluid_synth_sfload(synth.get(), "MuseScore_General.sf3", 1);
 
    auto sequencer = managed_ptr<fluid_sequencer_t>{ 
            new_fluid_sequencer2(0), delete_fluid_sequencer };
    fluid_seq_id_t synthSeqID = fluid_sequencer_register_fluidsynth(
            sequencer.get(), synth.get());
 
    auto renderer = managed_ptr<fluid_file_renderer_t>{ 
            new_fluid_file_renderer(synth.get()), delete_fluid_file_renderer };
 
// 총 128가지 악기에 대해, 38~87까지 50개 음의 소리를 생성합니다.
    for (size_t i = 0; i < limit; ++i)
    {
        for (size_t j = 0; j < 50; ++j)
        {
// 소리의 시작점
            size_t point = i * 50 * 2000 + j * 2000;
// 악기를 선택합니다.
            auto evt = managed_ptr<fluid_event_t>{ new_fluid_event(), delete_fluid_event };
            fluid_event_set_source(evt.get(), -1);
            fluid_event_set_dest(evt.get(), synthSeqID);
            fluid_event_program_change(evt.get(), 0, i);
            fluid_sequencer_send_at(sequencer.get(), evt.get(), 0 + point, true);
 
// 127의 세기로 38+j번째 음을 1000ms동안 재생합니다.
            evt = managed_ptr<fluid_event_t>{ new_fluid_event(), delete_fluid_event };
            fluid_event_set_source(evt.get(), -1);
            fluid_event_set_dest(evt.get(), synthSeqID);
            fluid_event_note(evt.get(), 0, 38 + j, 127, 1000);
            fluid_sequencer_send_at(sequencer.get(), evt.get(), 1 + point, true);
 
// 음 재생이 끝난 뒤에도 잔향(ASDR에서 release 부분)이 있을수 있으니
// 1999ms 후에는 모든 소리를 꺼줍니다.
            evt = managed_ptr<fluid_event_t>{ new_fluid_event(), delete_fluid_event };
            fluid_event_set_source(evt.get(), -1);
            fluid_event_set_dest(evt.get(), synthSeqID);
            fluid_event_all_sounds_off(evt.get(), 0);
            fluid_sequencer_send_at(sequencer.get(), evt.get(), 1999 + point, true);
        }
    }
 
// 타악기 소리 생성부분
    for (size_t j = 0; j < 46; ++j)
    {
// 소리의 시작점
        size_t point = limit * 50 * 2000 + j * 2000;
// 9번채널은 타악기 전용.
// 127의 세기로 1000ms 동안 j 타악기 음을 재생
        auto evt = managed_ptr<fluid_event_t>{ new_fluid_event(), delete_fluid_event };
        fluid_event_set_source(evt.get(), -1);
        fluid_event_set_dest(evt.get(), synthSeqID);
        fluid_event_note(evt.get(), 9, j + 34, 127, 1000);
        fluid_sequencer_send_at(sequencer.get(), evt.get(), 1 + point, true);
 
// 마찬가지로 1999ms 후에 모든 소리 꺼주기
        evt = managed_ptr<fluid_event_t>{ new_fluid_event(), delete_fluid_event };
        fluid_event_set_source(evt.get(), -1);
        fluid_event_set_dest(evt.get(), synthSeqID);
        fluid_event_all_sounds_off(evt.get(), 9);
        fluid_sequencer_send_at(sequencer.get(), evt.get(), 1999 + point, true);
    }
 
// 전체 시퀀서를 돌면서 실제 소리를 생성합니다.
    while (fluid_file_renderer_process_block(renderer.get()) == FLUID_OK)
    {
        if (fluid_sequencer_get_tick(sequencer.get()) > 2000 * (limit * 50 + 46)) break;
    }
    return 0;
}