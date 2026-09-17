/* Real-model regression for sampled BPE drift followed by two image turns.
 * Usage: test_server_vision_prefix MODEL MMPROJ [PREFILL_CHUNK]
 * Uses one engine, tiny sessions and no private conversation fixtures. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include <math.h>

static void expect(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); exit(1); }
}

static request image_request(server *s, int count, const char *suffix) {
    request req = {0};
    buf json = {0};
    buf_puts(&json, "[{\"role\":\"user\",\"content\":[");
    for (int i = 0; i < count; i++) {
        if (i) buf_puts(&json, ",");
        buf_printf(&json, "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/%s;base64,",
                   i ? "webp" : "png");
        buf_puts(&json, i ? test_inline_webp_base64 : test_inline_png_base64);
        buf_puts(&json, "\"}}");
    }
    buf_puts(&json, "]}]");
    const char *p = json.ptr;
    chat_msgs msgs = {0};
    expect(parse_messages(&p, &msgs), "parse image fixture");
    buf rendered = {0};
    buf_puts(&rendered, "<|im_start|>user\nfirst pressing ");
    for (int i = 0; i < count; i++) {
        if (i) buf_puts(&rendered, " after ");
        buf_puts(&rendered, msgs.v[0].images.v[i].marker);
    }
    buf_puts(&rendered, suffix);
    req.prompt_text = buf_take(&rendered);
    char err[256] = {0};
    expect(request_tokenize_multimodal_prompt(s->engine, s, &req, &msgs, err, sizeof(err)), err);
    chat_msgs_free(&msgs);
    buf_free(&json);
    return req;
}

static void free_image_request(request *req) {
    free(req->prompt_text);
    ds4_tokens_free(&req->prompt);
    for (size_t i = 0; i < req->image_count; i++)
        ds4_vision_embedding_free(&req->images[i].embedding);
    free(req->images);
    free(req->image_markers);
}

static void sync_images(server_slot *slot, request *req, ds4_tokens *tokens) {
    char err[256] = {0};
    expect(ds4_session_sync_multimodal(slot->session, tokens, req->images,
            req->image_count, err, sizeof(err)) == 0, err);
    expect(ds4_session_pos(slot->session) == tokens->len, "sync frontier");
}

static void check_rejected(server *s, server_slot *slot, request *req, const char *why) {
    ds4_tokens out = {0};
    uint32_t before[16];
    for (size_t i = 0; i < req->image_count; i++) before[i] = req->images[i].token_start;
    expect(live_text_prefix_prompt(s, slot, req, &out) == 0, why);
    for (size_t i = 0; i < req->image_count; i++)
        expect(req->images[i].token_start == before[i], "rejection leaves positions intact");
    expect(out.len == 0, "rejection leaves output intact");
    ds4_tokens_free(&out);
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL MMPROJ [PREFILL_CHUNK]\n", argv[0]);
        return 2;
    }
    int chunk = argc == 4 ? atoi(argv[3]) : 1;
    expect(chunk > 0 && chunk <= 2048, "prefill chunk must be 1..2048");
    server s = {0};
    pthread_mutex_init(&s.inference_mu, NULL);
    pthread_mutex_init(&s.model_mu, NULL);
    pthread_cond_init(&s.model_cv, NULL);
    ds4_engine_options opt = {.model_path = argv[1], .vision_path = argv[2],
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .context_size = 2048, .prefill_chunk = (uint32_t)chunk};
    expect(ds4_engine_open(&s.engine, &opt) == 0, "open engine");
    expect(ds4_engine_is_qwen4(s.engine), "Qwen fixture requires Qwen engine");
    server_slot slot = {0};
    expect(ds4_session_create(&slot.session, s.engine, 2048) == 0, "create session");
    ds4_tokens live = {0}, canonical = {0}, effective = {0};
    ds4_tokenize_rendered_chat(s.engine, "<|im_start|>user\nfirst press", &live);
    ds4_tokenize_rendered_chat(s.engine, "ing ", &live);
    ds4_tokenize_rendered_chat(s.engine, "<|im_start|>user\nfirst pressing ", &canonical);
    expect(live.len != canonical.len, "fixture must have BPE drift");
    char err[256] = {0};
    expect(ds4_session_sync(slot.session, &live, err, sizeof(err)) == 0, err);

    request first = image_request(&s, 1, " after ");
    int old = live.len;
    expect(live_text_prefix_prompt(&s, &slot, &first, &effective) == old,
           "text to first image reuses sampled tokens");
    expect(effective.v[old] == first.prompt.v[canonical.len], "keep Qwen vision-start token");
    expect(first.images[0].token_start == (uint32_t)old + 1, "image starts after wrapper");
    expect(effective.v[old + 1 + first.images[0].embedding.token_count] ==
           first.prompt.v[canonical.len + 1 + first.images[0].embedding.token_count],
           "keep Qwen vision-end token");
    sync_images(&slot, &first, &effective);
    int first_frontier = effective.len;
    expect(ds4_session_vision_image_count(slot.session) == 1, "one historical image");

    request second = image_request(&s, 2, " done");
    /* Raw pad-token rendering cannot match a request-local image marker:
     * this was the failing fallback before the fix. */
    size_t raw_len = 0;
    char *raw = render_tokens_text(s.engine, ds4_session_tokens(slot.session), &raw_len);
    expect(!byte_prefix_match(second.prompt_text, strlen(second.prompt_text), raw, raw_len),
           "fixture exercises old text fallback failure");
    free(raw);
    old = ds4_session_pos(slot.session);
    expect(ds4_session_common_prefix(slot.session, &second.prompt) < old,
           "fixture cannot hit token prefix");
    expect(!ds4_session_vision_prefix_matches(slot.session, second.images, 2),
           "canonical image position differs from live position");

    second.images[0].embedding.fingerprint[0] ^= 1;
    check_rejected(&s, &slot, &second, "changed historical image");
    second.images[0].embedding.fingerprint[0] ^= 1;
    second.images[0].embedding.token_count++;
    check_rejected(&s, &slot, &second, "changed historical row count");
    second.images[0].embedding.token_count--;
    char *press = strstr(second.prompt_text, "pressing");
    press[0] = 'd';
    check_rejected(&s, &slot, &second, "changed historical text");
    press[0] = 'p';
    char *saved = second.prompt_text;
    buf moved = {0};
    buf_puts(&moved, second.image_markers[0]);
    buf_puts(&moved, saved);
    second.prompt_text = moved.ptr;
    check_rejected(&s, &slot, &second, "moved image or literal marker cannot match");
    second.prompt_text = saved;
    buf_free(&moved);
    size_t count = second.image_count;
    second.image_count = 0;
    check_rejected(&s, &slot, &second, "removed images");
    second.image_count = count;

    expect(live_text_prefix_prompt(&s, &slot, &second, &effective) == old,
           "second image reuses the complete live frontier");
    expect(second.images[0].token_start == first.images[0].token_start,
           "historical image rebased to sampled frontier");
    expect(second.images[1].token_start == (uint32_t)old + 1, "new image retains wrapper");
    expect(!memcmp(effective.v, ds4_session_tokens(slot.session)->v, old * sizeof(int)),
           "exact live tokens retained");
    sync_images(&slot, &second, &effective);
    int predicted = ds4_session_argmax(slot.session);
    int vocab = ds4_engine_vocab_size(s.engine);
    float *cached_logits = xmalloc((size_t)vocab * sizeof(float));
    float *cold_logits = xmalloc((size_t)vocab * sizeof(float));
    expect(ds4_session_copy_logits(slot.session, cached_logits, vocab) == vocab, "copy cached logits");

    /* A fresh replay of the same effective tokens and conditioning must agree
     * with the cache continuation. Use sessions sequentially on one engine. */
    ds4_session_free(slot.session);
    expect(ds4_session_create(&slot.session, s.engine, 2048) == 0, "create cold session");
    sync_images(&slot, &second, &effective);
    int cold_predicted = ds4_session_argmax(slot.session);
    expect(ds4_session_copy_logits(slot.session, cold_logits, vocab) == vocab, "copy cold logits");
    float max_delta = 0;
    for (int i = 0; i < vocab; i++) {
        expect(isfinite(cached_logits[i]) && isfinite(cold_logits[i]), "finite logits");
        float d = fabsf(cached_logits[i] - cold_logits[i]);
        if (d > max_delta) max_delta = d;
    }
    printf("chunk=%d cached/cold logits max absolute delta: %.7g; next token: %d/%d\n",
           chunk, max_delta, predicted, cold_predicted);
    /* Qwen's MoE routing can amplify rounding changes between matrix shapes.
     * With chunk=1 every row uses the same shape, so cold replay is exact.
     * Larger chunks additionally require exact replay at the same frontiers,
     * matching the invariant in test_qwen4_prefill.c. */
    if (chunk == 1) {
        expect(max_delta == 0, "serial cold replay is byte-exact");
        expect(cold_predicted == predicted, "serial cold next token agrees");
    }
    ds4_session_invalidate(slot.session);
    expect(ds4_session_sync(slot.session, &live, err, sizeof(err)) == 0, err);
    ds4_tokens first_prefix = effective;
    first_prefix.len = first_frontier;
    sync_images(&slot, &first, &first_prefix);
    sync_images(&slot, &second, &effective);
    expect(ds4_session_copy_logits(slot.session, cold_logits, vocab) == vocab, "copy replay logits");
    expect(!memcmp(cached_logits, cold_logits, (size_t)vocab * sizeof(float)),
           "same-frontier replay is byte-exact");
    expect(ds4_session_argmax(slot.session) == predicted, "replayed next token agrees");
    free(cached_logits);
    free(cold_logits);

    request follow = image_request(&s, 2, " done plus");
    old = ds4_session_pos(slot.session);
    expect(live_text_prefix_prompt(&s, &slot, &follow, &effective) == old,
           "unchanged images and new nonces reuse cache");
    ds4_vision_span swap = follow.images[0];
    follow.images[0] = follow.images[1];
    follow.images[1] = swap;
    check_rejected(&s, &slot, &follow, "swapped historical images");
    follow.images[1].embedding.fingerprint[0] ^= 1;
    check_rejected(&s, &slot, &follow, "changed image identities");
    free_image_request(&follow);
    free_image_request(&second);
    free_image_request(&first);
    ds4_tokens_free(&effective);
    ds4_tokens_free(&canonical);
    ds4_tokens_free(&live);
    ds4_session_free(slot.session);
    server_image_cache_clear(&s.image_cache);
    ds4_engine_close(s.engine);
    puts("PASS: BPE drift, first/second image, wrappers, identity guards and cold replay");
    return 0;
}
