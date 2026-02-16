namespace Viktor
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using ProtoBuf;

    [ProtoContract]
    internal class Item
    {
        [ProtoMember(1)]
        public int Id { get; set; }
        
        [ProtoMember(2)]
        public string SemanticKey { get; set; } = string.Empty;
        
        [ProtoMember(3)]
        public string Payload { get; set; } = string.Empty;
    
        [ProtoMember(4)]
        public float[] Embedding { get; set; } = Array.Empty<float>();
    }
}

