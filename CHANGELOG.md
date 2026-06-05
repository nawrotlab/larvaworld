# Changelog

## v2.4.0 (2026-06-05)

### Fix

- Coerce nested classdict payloads ([`27e9017`](https://github.com/nawrotlab/larvaworld/commit/27e9017828675f9cb759568d35a6dbd0744a53a5))
- Return copied defaults from module inspector default_module_config ([`770a6a0`](https://github.com/nawrotlab/larvaworld/commit/770a6a0ff20253a79491281d7bc8b05cdfd4daa2))
- Keep module inspector sensor stimulus on active gain keys ([`5596365`](https://github.com/nawrotlab/larvaworld/commit/5596365b0f52de237531ca5388c7915fbc5616e5))

### Performance

- Reduce landing banner gif sizes ([`8ad8d3c`](https://github.com/nawrotlab/larvaworld/commit/8ad8d3c24b05dd54c15d8ddcff117ab7145edf7a))

### Documentation

- Keep module inspector copy user-facing ([`f2bfd7e`](https://github.com/nawrotlab/larvaworld/commit/f2bfd7e3d9ad94c51ac44613933ddfbf7e606289))

### Feature

- Extend module inspector with kind-aware feeder and sensor probes ([`51dc86b`](https://github.com/nawrotlab/larvaworld/commit/51dc86bd4f35b7085c76cf008eab022279a9b927))
- Add portal module inspector for crawler and turner ([`21f9e64`](https://github.com/nawrotlab/larvaworld/commit/21f9e64de6db5b01c1607e9a8735db11b62a8807))
- Refine model inspector preview reporters and mode widgets ([`3f9abb2`](https://github.com/nawrotlab/larvaworld/commit/3f9abb26a296854b206509ac9b6bc3d644889c97))
- Extend model inspector layout and live preview ([`19112ab`](https://github.com/nawrotlab/larvaworld/commit/19112ab56f81f669167186ae61e7756f2353e20c))

## v2.3.0 (2026-06-01)

### Fix

- Preserve dataset coordinate origins ([`97c7d7e`](https://github.com/nawrotlab/larvaworld/commit/97c7d7eba5f191dd29a8b6313a442304d3201b5d))
- Include replay arena metadata helpers ([`2413044`](https://github.com/nawrotlab/larvaworld/commit/24130444d1b0e8fdd2d61edf555bbd6911a9a188))
- Merge optional modules into model inspector summary ([`6720ac0`](https://github.com/nawrotlab/larvaworld/commit/6720ac06642f6221d11dd22d8dc88520c932364d))
- Make dataset replay sections boxes ([`41c189e`](https://github.com/nawrotlab/larvaworld/commit/41c189e3959865e1bbb45e117e7fbc82b792967c))
- Make dataset replay time range pygame-only ([`21486f8`](https://github.com/nawrotlab/larvaworld/commit/21486f8b3f6fc84b5d8a4f2c1e7508791da129f8))
- Disable unsupported native replay members ([`1858f79`](https://github.com/nawrotlab/larvaworld/commit/1858f794fe302319a1d4c8cdc1572c0c346f0e4f))
- Restore replay segment selection ([`ccdbe59`](https://github.com/nawrotlab/larvaworld/commit/ccdbe595b66c7fcd900d48f41c640e5d46570958))
- Restore larva-group dirty-state tracking ([`171c63b`](https://github.com/nawrotlab/larvaworld/commit/171c63b20e00868125f54b28a0e3353b20d18ba6))
- Stabilize replay and experiment summaries ([`6f5463c`](https://github.com/nawrotlab/larvaworld/commit/6f5463ce268012524b47da2d11d0a611a78fb873))
- Preserve dynamic template items on reload ([`8ee3941`](https://github.com/nawrotlab/larvaworld/commit/8ee39412de82734c9797b826b40614167f341cbd))
- Improve display shortcuts capture and popup styling ([`ef9c895`](https://github.com/nawrotlab/larvaworld/commit/ef9c895a93ce472c3bf6f4a16461ea973de87cf2))
- Improve single experiment status dialog formatting ([`7be5e10`](https://github.com/nawrotlab/larvaworld/commit/7be5e10175b5ec91a4dbb848bb4fbc3f686df076))
- Align circle envelope and allow legacy registry warnings ([`18298d9`](https://github.com/nawrotlab/larvaworld/commit/18298d901073158250c235b537798cb285b523c7))
- Stabilize workspace experiment template load/edit flow ([`e6adb61`](https://github.com/nawrotlab/larvaworld/commit/e6adb615f7780b456134324004811bd82225e17a))
- Disable head snapping by default in canvas preview ([`d1d2115`](https://github.com/nawrotlab/larvaworld/commit/d1d2115e4a30eb0f99c601c984757b941cfb8bbd))
- Stabilize experiment preset selection and runtime defaults ([`0e434b8`](https://github.com/nawrotlab/larvaworld/commit/0e434b880a140b78e69404311489a95e22844bce))
- Wire source group legend renderer ([`008a526`](https://github.com/nawrotlab/larvaworld/commit/008a5264102cdb6d0e14777ca37a8d41556e91fb))
- Harden intermitter state helpers ([`865293b`](https://github.com/nawrotlab/larvaworld/commit/865293b73b629b7838c9bdbf80a020fce5304c50))

### Refactor

- Arrange import environment controls ([`fb36f35`](https://github.com/nawrotlab/larvaworld/commit/fb36f354c9b09dd06a3d28b8a7fd63419533e936))
- Reuse replay param metadata ([`348d152`](https://github.com/nawrotlab/larvaworld/commit/348d1529ee08413792873489f2b9c6b15692463d))
- Share display shortcuts runtime helper ([`573ad0e`](https://github.com/nawrotlab/larvaworld/commit/573ad0ed1975789b2b40a82543582469415afc70))
- Remove legacy single experiment preview ([`2498a6e`](https://github.com/nawrotlab/larvaworld/commit/2498a6ef7cda735b9c0e81a110c5d9c75862f8c8))
- Improve single experiment parameters layout ([`aba7852`](https://github.com/nawrotlab/larvaworld/commit/aba7852a3b1bbefdf0cdacb9f71120dc74998877))

### Feature

- Refine import layout and replay support ([`fd6d66c`](https://github.com/nawrotlab/larvaworld/commit/fd6d66cd76c4490194410c5370327acbfcf3affa))
- Refine model inspector layout and validation ([`501edaa`](https://github.com/nawrotlab/larvaworld/commit/501edaa8223fbdc9f2091bed10571ed1301d5526))
- Finish model inspector persistence ([`6814144`](https://github.com/nawrotlab/larvaworld/commit/681414434b6a6245ad730822159ec06972f3f35f))
- Add native replay close inspection controls ([`e7a49e6`](https://github.com/nawrotlab/larvaworld/commit/e7a49e6ca60d25177523e5528c427e80b20803ae))
- Render larva contours and split canvas legends ([`13862c1`](https://github.com/nawrotlab/larvaworld/commit/13862c1d53f93c1a8570285aa0bff44fc8d8a0df))
- Draw larva body segments in simulation preview ([`78f8290`](https://github.com/nawrotlab/larvaworld/commit/78f8290ef42cf7a82b009219d4774e3d966c1057))
- Add agent index filtering and strict track-point selection ([`0410419`](https://github.com/nawrotlab/larvaworld/commit/041041908782b66e78e5f57262aa8c0310f0120b))
- Extend dataset replay for simulation runs ([`387eca0`](https://github.com/nawrotlab/larvaworld/commit/387eca042c864fc75f684de4ba1b4454ab45adf4))
- Make model inspector live-editable ([`752e014`](https://github.com/nawrotlab/larvaworld/commit/752e01453dddbc76ffe5c751aade87129b19ddd8))
- Migrate larva_models to parity-first model inspector ([`682ccee`](https://github.com/nawrotlab/larvaworld/commit/682cceec138ddff566556de357d9f95ee6d8a27c))
- Add dataset replay app and migrate track_viewer route ([`8386a85`](https://github.com/nawrotlab/larvaworld/commit/8386a85f1e309293c5a52f9b5b24b99527c3a8ae))
- Add workspace editable display shortcuts ([`9f4a28d`](https://github.com/nawrotlab/larvaworld/commit/9f4a28d23582083ebc1593e892cafab9fce66d51))
- Validate food source compatibility in single experiment ([`acc150e`](https://github.com/nawrotlab/larvaworld/commit/acc150eb38f6f00c7cdfad2b75056f17210224a9))
- Validate single experiment environment compatibility ([`1dea9e3`](https://github.com/nawrotlab/larvaworld/commit/1dea9e30fd081ea7ddacde76664742c3ee5584c2))
- Integrate experiment template preset controls ([`c6a0634`](https://github.com/nawrotlab/larvaworld/commit/c6a063418c761d727ce1beac4a83c5b4f5ad498a))
- Integrate env preset controls into single experiment ([`0e8107b`](https://github.com/nawrotlab/larvaworld/commit/0e8107b2512219a3ff76266aff9574f268fd0c19))
- Add generic registry/workspace preset controls ([`eb04124`](https://github.com/nawrotlab/larvaworld/commit/eb041241856ddf85fa08dbd350af4cf646161a1f))
- Remap workspace folders and refine single experiment ux ([`3f6d5c7`](https://github.com/nawrotlab/larvaworld/commit/3f6d5c765c3c5ca5f126c0bcf322cfa14f249704))
- Wire typed trials editor ([`7120876`](https://github.com/nawrotlab/larvaworld/commit/7120876f3554e51d7b1eedb44029f9763b060cbc))
- Wire typed collections editor ([`02bc538`](https://github.com/nawrotlab/larvaworld/commit/02bc538c3a6f8020ed4a5bbcee9305b529268bad))
- Wire typed simulation settings editor ([`a2a3e01`](https://github.com/nawrotlab/larvaworld/commit/a2a3e01ad6d9b3ca90d1fb6e51ae660ba8149884))
- Wire typed env params editor in single experiment ([`f464af7`](https://github.com/nawrotlab/larvaworld/commit/f464af700c26f3cd485b693c52030c83de1d993b))
- Wire typed enrichment editor in single experiment ([`8f78cd5`](https://github.com/nawrotlab/larvaworld/commit/8f78cd5a33121857020dc09bf1f619e76a9890ba))
- Wire typed larva_groups editor in single experiment ([`966ca36`](https://github.com/nawrotlab/larvaworld/commit/966ca36090d6d7bc0da47338441160bb45bbda7a))
- Add param-driven larva group helpers ([`dc08371`](https://github.com/nawrotlab/larvaworld/commit/dc0837102337e46a2dfdca7ba874f117b7ff1311))
- Refine single experiment preview visuals ([`e91afe8`](https://github.com/nawrotlab/larvaworld/commit/e91afe8121cbada386ced7b50ff3e393c69ee833))
- Polish single experiment preview controls ([`e5428d2`](https://github.com/nawrotlab/larvaworld/commit/e5428d253a8342dc88dec12ab33bef1b37c2e533))
- Add frame-based single experiment preview ([`7cc339c`](https://github.com/nawrotlab/larvaworld/commit/7cc339c0eaf779a418b55f5347b18b256f3138fa))
- Add larva preview frame capture helper ([`99feb21`](https://github.com/nawrotlab/larvaworld/commit/99feb21d877de9032d76d1c44151dd8777094359))
- Add simulated larvae canvas playback layer ([`c9e21dc`](https://github.com/nawrotlab/larvaworld/commit/c9e21dc8343099690e0665c95a4f9e538ad51c5b))
- Refine single experiment canvas preview layers ([`d67b48a`](https://github.com/nawrotlab/larvaworld/commit/d67b48a3a95091dda8d84db2f3a41908704324e8))
- Add shared environment canvas preview ([`d8f5622`](https://github.com/nawrotlab/larvaworld/commit/d8f5622490c5bfaca6d27b1b6c0019e2ea22740f))
- Refine import and simulation previews ([`bf3a8a9`](https://github.com/nawrotlab/larvaworld/commit/bf3a8a9a981ae64cd3250db2f6b75d44d34be8ac))

### Performance

- Make dataset replay geometry tick-local ([`d1afa8d`](https://github.com/nawrotlab/larvaworld/commit/d1afa8d32d0a8010162165c4a939d66fb5f5c46a))
- Optimize live display rendering ([`2a74b19`](https://github.com/nawrotlab/larvaworld/commit/2a74b19fd1d6be3521f7d34dce040b738dee0c06))

### Test

- Update preset integration assertions for tokenized flow ([`41e179a`](https://github.com/nawrotlab/larvaworld/commit/41e179a28f397f4a907ed7ed5f5d2b943a2748d1))
- Add env params typed roundtrip contracts ([`c112e1c`](https://github.com/nawrotlab/larvaworld/commit/c112e1c7554d9e09e0df4c062c352b69046c186f))
- Align environment preset labels with reverted behavior ([`a0ef347`](https://github.com/nawrotlab/larvaworld/commit/a0ef34732b5e9990668418ced3fb4a19fbc4fada))

### Style

- Apply ruff formatting to single experiment updates ([`19258f1`](https://github.com/nawrotlab/larvaworld/commit/19258f181ee5195d42dbae270e1949f999ed2781))

## v2.2.0 (2026-04-28)

### Test

- Remove unsupported ga conftype widget case ([`0b34ed7`](https://github.com/nawrotlab/larvaworld/commit/0b34ed71fd62020d8ea2be4a343d947999f114ee))
- Normalize import adapter path assertions ([`16ee822`](https://github.com/nawrotlab/larvaworld/commit/16ee82217a8deaf492d539aca88adec1cc7e424a))

### Documentation

- Update web applications guide ([`6600ef3`](https://github.com/nawrotlab/larvaworld/commit/6600ef3a874253a0f9a14fdfceac4f6ad80e686b))
- Update installation and optional deps ([`b286045`](https://github.com/nawrotlab/larvaworld/commit/b286045a815c4347962779ee686cc2902f6e52bd))

### Feature

- Integrate milestone m2 apps ([`be5a517`](https://github.com/nawrotlab/larvaworld/commit/be5a517fd282fc7280e555a4d2e24aa7e64c76d3))
- Add import config widget families ([`13bdb17`](https://github.com/nawrotlab/larvaworld/commit/13bdb17d74de7c2328523b8598e9e584de4e4c34))
- Add config widget helpers and demo app ([`b08ff7e`](https://github.com/nawrotlab/larvaworld/commit/b08ff7e61c2e04dcfc2180f7093da5586ad74eca))
- Add dataset manager and lane housing ([`2b2405f`](https://github.com/nawrotlab/larvaworld/commit/2b2405f51175b34e18513c7a71837f5389d4c4db))
- Refine dataset import app workflow ui ([`a78cf77`](https://github.com/nawrotlab/larvaworld/commit/a78cf7777ed3b97c842c6d395e06cfa4d4c65fa9))
- Share source directory picker ([`c1413e2`](https://github.com/nawrotlab/larvaworld/commit/c1413e211431f3c294bb26e0411ad8b55d103a76))
- Add experimental dataset import app ([`f99ff2e`](https://github.com/nawrotlab/larvaworld/commit/f99ff2e703687c5b724dd1dcf89b372a52a6bca4))
- Add workspace-first dataset adapters ([`8084330`](https://github.com/nawrotlab/larvaworld/commit/8084330884f3164e52cc4ac11a1944bda0bed4d1))
- Refine environment builder editor workflows ([`b8acd79`](https://github.com/nawrotlab/larvaworld/commit/b8acd79adacbad2247992f06c0a09351b50e500d))
- Harden environment builder presets and validation ([`b8aa71e`](https://github.com/nawrotlab/larvaworld/commit/b8aa71e878e231df80e0542ab52d7a364ac965fe))
- Refine environment builder and experiment ui ([`5351752`](https://github.com/nawrotlab/larvaworld/commit/5351752cdb68f8aa95f64d00f6f4ba9939557da7))
- Refine previews and environment editing ([`d1cd111`](https://github.com/nawrotlab/larvaworld/commit/d1cd111903c9e650d677f29edd5cec00b8d6461a))
- Add single experiment workflow ([`617ed36`](https://github.com/nawrotlab/larvaworld/commit/617ed3645468d3a2c730bc26b3430d5c7849828f))
- Expand environment builder workflow ([`cc96631`](https://github.com/nawrotlab/larvaworld/commit/cc96631b88eed10d7ea9c81cef255d7658ff7e33))
- Enforce workspace-first startup flow ([`9b5ffad`](https://github.com/nawrotlab/larvaworld/commit/9b5ffad9158e864a667e856913f2f8e006978e64))
- Add shared workspace management ([`b848c8e`](https://github.com/nawrotlab/larvaworld/commit/b848c8e683a66103c1041dd70d18b88654f4e719))
- Add gui_v2 desktop shell scaffold ([`82fc291`](https://github.com/nawrotlab/larvaworld/commit/82fc291c2ece855af532878fbf4a304bb92a9a03))
- Add rotating gif showcase banner on landing ([`e2ac5a1`](https://github.com/nawrotlab/larvaworld/commit/e2ac5a13af7618c4cf099a36bf18a983682cd0e0))
- Add quick-start modes and bootstrap loading flow ([`62ae008`](https://github.com/nawrotlab/larvaworld/commit/62ae008b08c63255bd14cd4af4627eddd78c41e3))
- Add environment builder app and startup loader ([`d38ea23`](https://github.com/nawrotlab/larvaworld/commit/d38ea23f20e36fd95a814c3ddbacc9053a9bbcc1))
- Remove demos lane and add persistent footer ([`b7f0eb3`](https://github.com/nawrotlab/larvaworld/commit/b7f0eb3327fa3b2db40ec1f532e20d77fd0bbbf9))
- Harden notebook launch flow and lane-styled notebook actions ([`db96de4`](https://github.com/nawrotlab/larvaworld/commit/db96de4d19a5876428f4ddd5a4456ff78fb655bf))
- Add tutorial notebook actions with workspace copies ([`0d28070`](https://github.com/nawrotlab/larvaworld/commit/0d28070ebecefaac7e9de0f412017355acdf4028))
- Add lane accents and stronger hover tint ([`f79ad47`](https://github.com/nawrotlab/larvaworld/commit/f79ad47c70e2f464965560fc76b727a1d7107d54))
- Add demo entrypoint and serve wiring ([`1cb02a7`](https://github.com/nawrotlab/larvaworld/commit/1cb02a7e3e152f6b996cd5bcd42a3dad33342a9e))
- Add landing and preview apps ([`b4c2a77`](https://github.com/nawrotlab/larvaworld/commit/b4c2a778f598e6cae24ab1b72b478b44e0f8a5c2))
- Add registry core and smoke tests ([`57b1ad8`](https://github.com/nawrotlab/larvaworld/commit/57b1ad828215db6cebc0fee2aae391c8481bf6af))

### Fix

- Reset lab format registry from import app ([`1206931`](https://github.com/nawrotlab/larvaworld/commit/1206931894fbfa9e35d1e41660cca3dee419e931))
- Stabilize import environment and tracker config ([`72d1b4a`](https://github.com/nawrotlab/larvaworld/commit/72d1b4ad6250a79756acd07d6a32e91e750c9be8))
- Stabilize portal regressions and arena edge cases ([`c86cdaf`](https://github.com/nawrotlab/larvaworld/commit/c86cdafd4745d8d98b63bf00707761e46fbf3b9f))
- Refine environment builder interactions ([`5faab32`](https://github.com/nawrotlab/larvaworld/commit/5faab32f833a40d68c51fee68a810820cf89b585))
- Correct quick-start tab layering and active styling ([`611305b`](https://github.com/nawrotlab/larvaworld/commit/611305bf374d20b40d45d3b4f0d377f1ea7e1d89))
- Restore full-tile click overlay navigation ([`dedc2ba`](https://github.com/nawrotlab/larvaworld/commit/dedc2ba7c279dab9115466b3336d12aa9f0150d7))
- Enforce unique lane membership ([`062be56`](https://github.com/nawrotlab/larvaworld/commit/062be564a01080190e31cf7a872ef20ceecacdec))

### Refactor

- Refine environment builder and portal cleanup ([`1e22ccd`](https://github.com/nawrotlab/larvaworld/commit/1e22ccd217325949a219c9e2c26417f9030ba04d))
- Remove demo mode and preview route ([`c7aa023`](https://github.com/nawrotlab/larvaworld/commit/c7aa023962637bd559ea8695515441a3438fce3d))

### Build

- Refresh poetry lockfile ([`ecd1546`](https://github.com/nawrotlab/larvaworld/commit/ecd1546d9979152a2b758a91c3918c920a39640c))

### Style

- Polish header layout and settings dropdown ([`abd3573`](https://github.com/nawrotlab/larvaworld/commit/abd35730f5f7d976fc5992c92ee9603ead8e06fa))
- Keep grid layout panel-controlled ([`adf01de`](https://github.com/nawrotlab/larvaworld/commit/adf01de6c8e6c47ee25e31859769769a41f372e1))

## v2.1.1 (2026-01-13)

### Documentation

- Update tutorials and visualization guides ([`89af527`](https://github.com/nawrotlab/larvaworld/commit/89af527dae6c6824200f36f1647e0d2b78bf63a8))
- Align docs examples with v2.1.0 api ([`8030d2c`](https://github.com/nawrotlab/larvaworld/commit/8030d2caeaf50d43cf80a21384d0f4ddf0ff7e98))

### Fix

- Sync examples with code and harden eval plots ([`c6b8cc5`](https://github.com/nawrotlab/larvaworld/commit/c6b8cc5adb89985d842e3ead23863de078a43e56))

## v2.1.0 (2025-12-21)

### Documentation

- Add summary of all pr-4c changes to unreleased section ([`c47fcdc`](https://github.com/nawrotlab/larvaworld/commit/c47fcdc7e5eb2b2811f74ac897009b46c366ad7d))
- Add type examples and improve commit message documentation ([`bda3604`](https://github.com/nawrotlab/larvaworld/commit/bda3604831fc411b6a639a25459cbead3cced4ea))
- Remove codecov, poetry, and ruff badges from readme ([`7576a79`](https://github.com/nawrotlab/larvaworld/commit/7576a79e7e3b6f0ade06bdfb5d4ff3d4a16c73ae))
- Fix module-level constant docstrings for autoapi ([`8911126`](https://github.com/nawrotlab/larvaworld/commit/8911126ba30ffb89da0af07d73b33bb5860432f5))
- Update first publication link in publications page ([`6972912`](https://github.com/nawrotlab/larvaworld/commit/6972912dc787e25791a2ace39cce4992bea1438a))
- Add publications page and clarify cli argument order ([`d7cf5c2`](https://github.com/nawrotlab/larvaworld/commit/d7cf5c2a08765d03201ac8bf07f0c3cb8efc8f7b))

### Feature

- Python 3.12 &amp; 3.13 support, collision handling fixes, and code cleanup ([`0842aaf`](https://github.com/nawrotlab/larvaworld/commit/0842aafab994111d3db825243e9704ddfe8acb8d))
- Add storage directory feedback and update python 3.10-3.13 docs ([`e5e975b`](https://github.com/nawrotlab/larvaworld/commit/e5e975bbec53cfa08865f59637543242de9f31f2))

### Build

- Add imageio[ffmpeg] extra and use reg.default_refid ([`45897f1`](https://github.com/nawrotlab/larvaworld/commit/45897f1485bff45b0ee02114f7e71248cc5500d7))

### Fix

- Panel compatibility and add python 3.13 support ([`68cc0d0`](https://github.com/nawrotlab/larvaworld/commit/68cc0d0660d58f4a9b0766c97c0a7b34e9ee1658))
- Python 3.12 support and collision handling fixes ([`e1da5fc`](https://github.com/nawrotlab/larvaworld/commit/e1da5fc9da4dc0cd40e838c1522a4acab636d39a))
- Align timer baseline across time components ([`299eb59`](https://github.com/nawrotlab/larvaworld/commit/299eb59ecd5cf0b2090321ebbe4ca1ca3c0ef3db))

### Refactor

- Remove deprecation warnings and strict import checks ([`2e80eaf`](https://github.com/nawrotlab/larvaworld/commit/2e80eafe9d62a431d947997f4ec1e712c64acc6b))

## v2.0.1 (2025-11-25)

### Fix

- Improve installation docs, ci workflow, and simulation handling ([`670f66b`](https://github.com/nawrotlab/larvaworld/commit/670f66b74c37cfd5a20d3e29534b7aef88302592))
- Improve simulation window handling and pause feedback ([`304ce80`](https://github.com/nawrotlab/larvaworld/commit/304ce807e3ae5f202653b3ce24ce9568e9f15286))
- Properly detect linting errors vs formatting changes ([`679821c`](https://github.com/nawrotlab/larvaworld/commit/679821cfd268c9732a276d05cce04661750ee53f))
- Correct has_changes variable check in lint job ([`484bb00`](https://github.com/nawrotlab/larvaworld/commit/484bb0041fcb941fbaaaf800854c4ef1fefb8b0f))
- Broken documentation references to tutorials/index ([`6b3a41c`](https://github.com/nawrotlab/larvaworld/commit/6b3a41c6f815f7d46c9f5eb1b647771256ae1c9e))
- Simulation termination and visualization documentation ([`6b56f91`](https://github.com/nawrotlab/larvaworld/commit/6b56f91a3f316739e828ad5bb91f0eae6c75db38))

### Documentation

- Expand video examples ([`f13ee97`](https://github.com/nawrotlab/larvaworld/commit/f13ee97575a02f93dd4bf4ad2a7c6ca7fe18ba04))
- Hide tutorials toctree from main page, keep in sidebar ([`c9537ec`](https://github.com/nawrotlab/larvaworld/commit/c9537ec6cd622c694243b0aac925e1a3f569ac73))
- Remove :hidden: from tutorials toctree to show in sidebar ([`a473fdd`](https://github.com/nawrotlab/larvaworld/commit/a473fddb298aae15fcbdd827173e8bf4a28e98df))
- Restore tutorial subsections structure with .rst files ([`8aad1a5`](https://github.com/nawrotlab/larvaworld/commit/8aad1a5bb3b532b50d66ce8709fb51bcf564f9c3))
- Create tutorial subsections with index files (configuration, simulation, data, development) ([`a27d098`](https://github.com/nawrotlab/larvaworld/commit/a27d098dde933ec959e0ccabb945d4f2bdfade10))
- Organize tutorials into subsections (configuration, simulation, data, development) ([`6f60d95`](https://github.com/nawrotlab/larvaworld/commit/6f60d958b302c50ffdbae3dbae361cfba0819338))
- Remove duplicate myst_parser extension (included in myst_nb) ([`c2e63d1`](https://github.com/nawrotlab/larvaworld/commit/c2e63d1c1380d15a74e5e31c5c6278f2952b3de5))
- Switch from nbsphinx to myst_nb and add pygments style ([`76ffea2`](https://github.com/nawrotlab/larvaworld/commit/76ffea2de6bfd9046f14b21592986d2797bc0f40))
- Switch to sphinx_rtd_theme and rename autoapi entry ([`e92eac0`](https://github.com/nawrotlab/larvaworld/commit/e92eac0062d325426c5daf141e570a6f7e1caf3c))
- Fix sidebar navigation and use default furo theme ([`6597030`](https://github.com/nawrotlab/larvaworld/commit/659703028960614eb45c47d6c06401c97bc38767))
- Reorganize concepts and index ([`4d6a563`](https://github.com/nawrotlab/larvaworld/commit/4d6a56347080b4afed96ec63d64b87491f4ce016))

### Refactor

- Documentation improvements, ci enhancements, and test marker refactoring ([`838b554`](https://github.com/nawrotlab/larvaworld/commit/838b55431584b9aa9c0ba18ce89a87fe79c09b0d))
- Rename pytest marker from &#39;slow&#39; to &#39;heavy&#39; ([`333de3c`](https://github.com/nawrotlab/larvaworld/commit/333de3cfd726c6edfb05c67f494e53baacc673e9))

### Build

- Refresh poetry.lock ([`f7c491b`](https://github.com/nawrotlab/larvaworld/commit/f7c491b92806c75b132726d2170970f14e0e7e1e))
- Refresh poetry.lock ([`37e167f`](https://github.com/nawrotlab/larvaworld/commit/37e167fa2ce68f7605a5a525ea14387541daddce))

## v2.0.0 (2025-11-22)

### Style

- Apply pre-commit formatting fixes ([`b2b9071`](https://github.com/nawrotlab/larvaworld/commit/b2b9071e674536b9b08c540733edaf5febd28400))

### Fix

- Update poetry.lock to include sphinxcontrib-mermaid ([`ac9eb33`](https://github.com/nawrotlab/larvaworld/commit/ac9eb33869d3814105e467fbfabdc5d860cd940a))

### Documentation

- Major documentation overhaul with sphinx/readthedocs setup ([`e453018`](https://github.com/nawrotlab/larvaworld/commit/e453018e6576c1806ba7294ea044d5ac1baccb59))
- Update license to mit and fix python version constraints ([`8f136f8`](https://github.com/nawrotlab/larvaworld/commit/8f136f8efb889a4bed72ef6816a3c9f797373851))

### Breaking

- Complete package modernization (phases 1-4) (#3) ([`188bfb7`](https://github.com/nawrotlab/larvaworld/commit/188bfb7643c3c33fb62d8af52ba033473c5984a5))

## v1.0.0 (2025-05-08)

### Feature

- Start semver at 1.0.0 (#35) ([`f532b66`](https://github.com/nawrotlab/larvaworld/commit/f532b6653ad0a5bba8111194c99ec87f2e7e3efe))

## v0.1.0 (2025-05-08)

### Fix

- Semantic versioning ([`a224d62`](https://github.com/nawrotlab/larvaworld/commit/a224d62c2792ec195a8c95885b0f82f10d9f0c4e))
- Semantic versioning ([`a6c3929`](https://github.com/nawrotlab/larvaworld/commit/a6c3929f2a588fdee0e8ba90f8ff0537f0ba37b4))

## v0.1.0-rc.1 (2025-04-22)

### Fix

- Run venv test only on linux ([`cb2bcee`](https://github.com/nawrotlab/larvaworld/commit/cb2bcee20be0205db39d88bb0b8750d3c9e2fed8))
- Remove importlib dependency ([`a065475`](https://github.com/nawrotlab/larvaworld/commit/a06547572c2fac881e16172519c1b6ac2339e1a2))
- Add missing docopts dependency ([`248611c`](https://github.com/nawrotlab/larvaworld/commit/248611cc3fc478cabd93d0059c474ef96741645b))

### Feature

- Add venv install test to github action ([`edd69f5`](https://github.com/nawrotlab/larvaworld/commit/edd69f503754dda864236356cda3cc7c4cc06b49))
- Add venv install test to github action ([`09f218f`](https://github.com/nawrotlab/larvaworld/commit/09f218f14b4b9fc65e1fe152604facc7a81099bf))
- Add example code for remote brian interface and tutorial notebook ([`00e0b0c`](https://github.com/nawrotlab/larvaworld/commit/00e0b0ca88c0a21f099e048c30dd0a3feeec15bc))
- Add tutorial notebooks on library interface and custom modules ([`ec1dbd5`](https://github.com/nawrotlab/larvaworld/commit/ec1dbd5cd2c41af9f9fea01dac2ca76dd9dfccca))

## v0.0.1-rc.1 (2024-11-24)

### Fix

- Use master instead of main branch ([`a1d054b`](https://github.com/nawrotlab/larvaworld/commit/a1d054ba24ea5c1c8dab525a6b45be3678cbde47))
